import json
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from kubernetes.client.exceptions import ApiException

from agent.tools import kubernetes_tools as kt


def ns(**kwargs):
    return SimpleNamespace(**kwargs)


def metadata(name="sample", namespace="mlops-a03"):
    return ns(
        name=name,
        namespace=namespace,
        uid="test-uid",
        labels={"app": "inference-api"},
        creation_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        generation=1,
    )


def make_pod(
    name="sample-pod",
    phase="Running",
    ready=True,
    container_state="running",
):
    waiting = ns(reason="ImagePullBackOff", message="Image unavailable")
    terminated = ns(reason="Error", message="Container exited", exit_code=1)
    running = ns()
    state = ns(
        waiting=waiting if container_state == "waiting" else None,
        terminated=terminated if container_state == "terminated" else None,
        running=running if container_state == "running" else None,
    )
    container = ns(
        name="inference-api",
        ready=ready,
        restart_count=0,
        image="mlops-a03-inference:0.1.0",
        image_id="sha256:test",
        state=state,
    )
    condition = ns(
        type="Ready",
        status="True" if ready else "False",
        reason=None,
        message=None,
        last_transition_time=None,
    )
    return ns(
        metadata=metadata(name),
        status=ns(
            phase=phase,
            pod_ip="10.0.0.10",
            container_statuses=[container],
            conditions=[condition],
        ),
        spec=ns(node_name="minikube"),
    )


def make_deployment():
    return ns(
        metadata=metadata("inference-api"),
        status=ns(
            ready_replicas=2,
            available_replicas=2,
            updated_replicas=2,
            unavailable_replicas=0,
            observed_generation=1,
            conditions=[],
        ),
        spec=ns(
            replicas=2,
            selector=ns(match_labels={"app": "inference-api"}),
            template=ns(
                spec=ns(
                    containers=[
                        ns(image="mlops-a03-inference:0.1.0")
                    ]
                )
            ),
        ),
    )


def make_service():
    return ns(
        metadata=metadata("inference-api"),
        spec=ns(
            type="ClusterIP",
            cluster_ip="10.111.149.213",
            external_ips=["192.0.2.10"],
            selector={"app": "inference-api"},
            ports=[
                ns(
                    name="http",
                    protocol="TCP",
                    port=8000,
                    target_port=8000,
                    node_port=None,
                )
            ],
        ),
    )


@pytest.fixture
def clients(monkeypatch):
    core = Mock()
    apps = Mock()
    monkeypatch.setattr(kt, "_get_clients", lambda: (core, apps))
    return core, apps


def test_rejects_unapproved_namespace_before_api_call(monkeypatch):
    get_clients = Mock()
    monkeypatch.setattr(kt, "_get_clients", get_clients)

    result = kt.list_pods(namespace="default")

    assert result["success"] is False
    get_clients.assert_not_called()


@pytest.mark.parametrize(
    "name",
    ["", "../pod", "bad/name", "name with spaces"],
)
def test_rejects_invalid_pod_names_before_api_call(clients, name):
    core, _ = clients

    result = kt.get_pod_details(name=name)

    assert result["success"] is False
    core.read_namespaced_pod.assert_not_called()


@pytest.mark.parametrize("limit", [0, -1, 101, True])
def test_rejects_invalid_pagination_limit(clients, limit):
    core, _ = clients

    result = kt.list_pods(limit=limit)

    assert result["success"] is False
    core.list_namespaced_pod.assert_not_called()


def test_list_pods_serializes_results_and_pagination(clients):
    core, _ = clients
    core.list_namespaced_pod.return_value = ns(
        items=[make_pod()],
        metadata=ns(_continue="next-page"),
    )

    result = kt.list_pods(limit=10, continue_token="previous-page")

    assert result["success"] is True
    assert result["count"] == 1
    assert result["continue_token"] == "next-page"
    assert result["data"][0]["name"] == "sample-pod"
    assert result["data"][0]["phase"] == "Running"
    assert result["data"][0]["ready"] is True
    core.list_namespaced_pod.assert_called_once_with(
        namespace="mlops-a03",
        limit=10,
        _continue="previous-page",
    )
    json.dumps(result)


def test_list_pods_handles_empty_result(clients):
    core, _ = clients
    core.list_namespaced_pod.return_value = ns(
        items=[],
        metadata=ns(_continue=None),
    )

    result = kt.list_pods()

    assert result["success"] is True
    assert result["data"] == []
    assert result["count"] == 0


def test_get_pod_details_serializes_container_state(clients):
    core, _ = clients
    core.read_namespaced_pod.return_value = make_pod(
        container_state="waiting",
        ready=False,
    )

    result = kt.get_pod_details("sample-pod")

    assert result["success"] is True
    assert result["data"]["container_statuses"][0]["state"] == "waiting"
    assert (
        result["data"]["container_statuses"][0]["reason"]
        == "ImagePullBackOff"
    )
    json.dumps(result)


def test_list_deployments_serializes_replicas(clients):
    _, apps = clients
    apps.list_namespaced_deployment.return_value = ns(
        items=[make_deployment()],
        metadata=ns(_continue=None),
    )

    result = kt.list_deployments()

    assert result["success"] is True
    deployment = result["data"][0]
    assert deployment["desired_replicas"] == 2
    assert deployment["ready_replicas"] == 2
    assert deployment["images"] == ["mlops-a03-inference:0.1.0"]
    json.dumps(result)


def test_list_services_serializes_external_ips_and_ports(clients):
    core, _ = clients
    core.list_namespaced_service.return_value = ns(
        items=[make_service()],
        metadata=ns(_continue=None),
    )

    result = kt.list_services()

    assert result["success"] is True
    service = result["data"][0]
    assert service["external_ips"] == ["192.0.2.10"]
    assert service["ports"][0]["port"] == 8000
    assert service["cluster_ip"] == "10.111.149.213"
    json.dumps(result)


def test_get_pod_details_returns_structured_not_found_error(clients):
    core, _ = clients
    core.read_namespaced_pod.side_effect = ApiException(
        status=404,
        reason="Not Found",
    )

    result = kt.get_pod_details("missing-pod")

    assert result["success"] is False
    assert result["status_code"] == 404


def test_get_pod_logs_uses_bounded_tail_and_no_follow(clients):
    core, _ = clients
    core.read_namespaced_pod_log.return_value = "container started"

    result = kt.get_pod_logs(
        "sample-pod",
        container="inference-api",
        tail_lines=25,
        timestamps=True,
    )

    assert result["success"] is True
    core.read_namespaced_pod_log.assert_called_once_with(
        name="sample-pod",
        namespace="mlops-a03",
        container="inference-api",
        tail_lines=25,
        timestamps=True,
        limit_bytes=1024 * 1024,
        follow=False,
    )
    json.dumps(result)


def test_get_pod_logs_rejects_excessive_tail_lines(clients):
    core, _ = clients

    result = kt.get_pod_logs("sample-pod", tail_lines=1001)

    assert result["success"] is False
    core.read_namespaced_pod_log.assert_not_called()


def test_list_events_returns_json_compatible_result(clients):
    core, _ = clients
    event = ns(
        metadata=metadata("sample-event"),
        type="Warning",
        reason="BackOff",
        message="Container restarting",
        involved_object=ns(
            kind="Pod",
            name="sample-pod",
            namespace="mlops-a03",
        ),
        first_timestamp=None,
        last_timestamp=None,
        count=2,
        source=ns(component="kubelet", host="minikube"),
    )
    core.list_namespaced_event.return_value = ns(
        items=[event],
        metadata=ns(_continue=None),
    )

    result = kt.list_events()

    assert result["success"] is True
    assert result["count"] == 1
    json.dumps(result)


def test_diagnose_pod_returns_json_compatible_result(clients):
    core, _ = clients
    core.read_namespaced_pod.return_value = make_pod()

    result = kt.diagnose_pod("sample-pod")

    assert result["success"] is True
    assert isinstance(result["data"], dict)
    json.dumps(result)


def test_diagnose_unhealthy_pod_returns_json_compatible_result(clients):
    core, _ = clients
    core.read_namespaced_pod.return_value = make_pod(
        phase="Pending",
        ready=False,
        container_state="waiting",
    )

    result = kt.diagnose_pod("sample-pod")

    assert result["success"] is True
    assert isinstance(result["data"], dict)
    json.dumps(result)


# Configuration and error-handling coverage

def test_get_clients_falls_back_to_incluster_config(monkeypatch):
    core = Mock()
    apps = Mock()
    load_kube = Mock(side_effect=kt.config.ConfigException("not configured"))
    load_cluster = Mock()

    monkeypatch.setattr(kt.config, "load_kube_config", load_kube)
    monkeypatch.setattr(kt.config, "load_incluster_config", load_cluster)
    monkeypatch.setattr(kt.client, "CoreV1Api", lambda: core)
    monkeypatch.setattr(kt.client, "AppsV1Api", lambda: apps)

    result = kt._get_clients()

    assert result == (core, apps)
    load_kube.assert_called_once()
    load_cluster.assert_called_once()


def test_get_clients_raises_safe_runtime_error_when_both_configs_fail(
    monkeypatch,
):
    monkeypatch.setattr(
        kt.config,
        "load_kube_config",
        Mock(side_effect=kt.config.ConfigException("local config missing")),
    )
    monkeypatch.setattr(
        kt.config,
        "load_incluster_config",
        Mock(side_effect=kt.config.ConfigException("cluster config missing")),
    )

    with pytest.raises(RuntimeError, match="Kubernetes configuration is unavailable"):
        kt._get_clients()


@pytest.mark.parametrize(
    ("exception", "expected_status", "expected_error"),
    [
        (
            ApiException(status=403, reason="Forbidden"),
            403,
            "Kubernetes API access is forbidden",
        ),
        (
            ApiException(status=500, reason="Internal Server Error"),
            500,
            "Kubernetes API request failed",
        ),
    ],
)
def test_api_errors_are_sanitized(clients, exception, expected_status, expected_error):
    core, _ = clients
    core.list_namespaced_pod.side_effect = exception

    result = kt.list_pods()

    assert result == {
        "success": False,
        "error": expected_error,
        "status_code": expected_status,
    }


def test_runtime_error_is_returned_as_safe_message(clients):
    core, _ = clients
    core.list_namespaced_pod.side_effect = RuntimeError(
        "Kubernetes configuration is unavailable"
    )

    result = kt.list_pods()

    assert result == {
        "success": False,
        "error": "Kubernetes configuration is unavailable",
    }


def test_unexpected_programming_error_is_not_hidden(clients):
    core, _ = clients
    core.list_namespaced_pod.side_effect = ValueError("programming defect")

    with pytest.raises(ValueError, match="programming defect"):
        kt.list_pods()


# Deployment and Service detail operations

def test_get_deployment_details_serializes_deployment(clients):
    _, apps = clients
    apps.read_namespaced_deployment.return_value = make_deployment()

    result = kt.get_deployment_details("inference-api")

    assert result["success"] is True
    assert result["data"]["desired_replicas"] == 2
    assert result["data"]["available_replicas"] == 2
    json.dumps(result)


def test_get_service_details_serializes_service(clients):
    core, _ = clients
    core.read_namespaced_service.return_value = make_service()

    result = kt.get_service_details("inference-api")

    assert result["success"] is True
    assert result["data"]["type"] == "ClusterIP"
    assert result["data"]["external_ips"] == ["192.0.2.10"]
    json.dumps(result)


def test_get_service_details_handles_forbidden_api_access(clients):
    core, _ = clients
    core.read_namespaced_service.side_effect = ApiException(
        status=403,
        reason="Forbidden",
    )

    result = kt.get_service_details("inference-api")

    assert result["success"] is False
    assert result["status_code"] == 403
    assert result["error"] == "Kubernetes API access is forbidden"


# Input validation

@pytest.mark.parametrize("container", ["bad/name", "bad name", "", "../container"])
def test_get_pod_logs_rejects_invalid_container_names(clients, container):
    core, _ = clients

    result = kt.get_pod_logs("sample-pod", container=container)

    assert result["success"] is False
    assert result["error"] == "Invalid container name"
    core.read_namespaced_pod_log.assert_not_called()


@pytest.mark.parametrize("timestamps", ["true", 1, None])
def test_get_pod_logs_rejects_non_boolean_timestamps(clients, timestamps):
    core, _ = clients

    result = kt.get_pod_logs("sample-pod", timestamps=timestamps)

    assert result["success"] is False
    assert result["error"] == "Invalid timestamps"
    core.read_namespaced_pod_log.assert_not_called()


def test_list_events_forwards_pagination(clients):
    core, _ = clients
    core.list_namespaced_event.return_value = ns(
        items=[],
        metadata=ns(_continue="events-next"),
    )

    result = kt.list_events(limit=12, continue_token="events-previous")

    assert result["success"] is True
    assert result["data"] == []
    assert result["continue_token"] == "events-next"
    core.list_namespaced_event.assert_called_once_with(
        namespace="mlops-a03",
        limit=12,
        _continue="events-previous",
    )


def test_list_events_sanitizes_api_failure(clients):
    core, _ = clients
    core.list_namespaced_event.side_effect = ApiException(
        status=403,
        reason="Forbidden",
    )

    result = kt.list_events()

    assert result["success"] is False
    assert result["status_code"] == 403


# Pod diagnosis outcomes

def test_diagnose_healthy_running_pod(clients):
    core, _ = clients
    core.read_namespaced_pod.return_value = make_pod()

    result = kt.diagnose_pod("sample-pod")

    assert result["success"] is True
    assert result["data"]["health"] == "healthy"
    assert result["data"]["summary"] == "Pod is running and ready"
    assert result["data"]["issues"] == []


def test_diagnose_failed_pod(clients):
    core, _ = clients
    pod = make_pod(phase="Failed", ready=False, container_state="terminated")
    core.read_namespaced_pod.return_value = pod

    result = kt.diagnose_pod("sample-pod")

    assert result["success"] is True
    assert result["data"]["health"] == "unhealthy"
    assert "Pod phase is Failed" in result["data"]["issues"]
    assert "Container inference-api exited with code 1" in result["data"]["issues"]


def test_diagnose_unready_pod_without_other_issues(clients):
    core, _ = clients
    pod = make_pod(phase="Running", ready=False)
    pod.status.container_statuses[0].state.waiting = None
    pod.status.container_statuses[0].state.terminated = None
    pod.status.container_statuses[0].state.running = ns()
    core.read_namespaced_pod.return_value = pod

    result = kt.diagnose_pod("sample-pod")

    assert result["data"]["health"] == "unhealthy"
    assert result["data"]["issues"] == ["Pod is not Ready"]


def test_diagnose_unknown_phase(clients):
    core, _ = clients
    pod = make_pod(phase="Succeeded", ready=True)
    core.read_namespaced_pod.return_value = pod

    result = kt.diagnose_pod("sample-pod")

    assert result["success"] is True
    assert result["data"]["health"] == "unknown"
    assert result["data"]["summary"] == "Pod phase is Succeeded"


def test_diagnose_propagates_pod_lookup_failure(clients):
    core, _ = clients
    core.read_namespaced_pod.side_effect = ApiException(
        status=404,
        reason="Not Found",
    )

    result = kt.diagnose_pod("missing-pod")

    assert result["success"] is False
    assert result["status_code"] == 404


# Additional validation and client-failure coverage

def test_rejects_malformed_namespace_before_api_call(monkeypatch):
    get_clients = Mock()
    monkeypatch.setattr(kt, "_get_clients", get_clients)

    result = kt.list_pods(namespace=None)

    assert result == {"success": False, "error": "Invalid namespace"}
    get_clients.assert_not_called()


@pytest.mark.parametrize(
    "token",
    ["", 123, "invalid\n token", "x" * (kt.MAX_CONTINUE_TOKEN_LENGTH + 1)],
)
def test_rejects_invalid_continuation_tokens(clients, token):
    core, _ = clients

    result = kt.list_pods(continue_token=token)

    assert result == {
        "success": False,
        "error": "Invalid continuation token",
    }
    core.list_namespaced_pod.assert_not_called()


@pytest.mark.parametrize(
    ("operation", "kwargs"),
    [
        (kt.list_deployments, {"namespace": "default"}),
        (kt.list_events, {"namespace": "default"}),
    ],
)
def test_list_operations_reject_unapproved_namespace(
    clients, operation, kwargs
):
    core, apps = clients

    result = operation(**kwargs)

    assert result == {
        "success": False,
        "error": "Namespace is not approved",
    }
    core.assert_not_called()
    apps.assert_not_called()


@pytest.mark.parametrize(
    ("operation", "kwargs", "expected_error"),
    [
        (
            kt.list_deployments,
            {"continue_token": ""},
            "Invalid continuation token",
        ),
        (
            kt.list_events,
            {"continue_token": ""},
            "Invalid continuation token",
        ),
        (
            kt.get_deployment_details,
            {"name": "bad/name"},
            "Invalid deployment name",
        ),
        (
            kt.get_service_details,
            {"name": "bad/name"},
            "Invalid service name",
        ),
        (
            kt.get_pod_logs,
            {"name": "bad/name"},
            "Invalid pod name",
        ),
    ],
)
def test_operations_reject_invalid_inputs_before_api_call(
    clients, operation, kwargs, expected_error
):
    core, apps = clients

    result = operation(**kwargs)

    assert result == {"success": False, "error": expected_error}
    core.assert_not_called()
    apps.assert_not_called()


def test_get_deployment_details_sanitizes_api_failure(clients):
    _, apps = clients
    apps.read_namespaced_deployment.side_effect = ApiException(
        status=500,
        reason="Internal Server Error",
    )

    result = kt.get_deployment_details("inference-api")

    assert result == {
        "success": False,
        "error": "Kubernetes API request failed",
        "status_code": 500,
    }


def test_kubernetes_client_os_error_uses_safe_generic_error(clients):
    core, _ = clients
    core.list_namespaced_pod.side_effect = OSError("sensitive local detail")

    result = kt.list_pods()

    assert result == {
        "success": False,
        "error": "Kubernetes client failed: OSError",
    }
    assert "sensitive local detail" not in str(result)


@pytest.mark.parametrize(
    "operation,kwargs",
    [
        (kt.get_pod_details, {"name": "sample-pod"}),
        (kt.get_deployment_details, {"name": "inference-api"}),
        (kt.get_service_details, {"name": "inference-api"}),
        (kt.get_pod_logs, {"name": "sample-pod"}),
    ],
)
def test_detail_operations_reject_malformed_namespace(
    clients, operation, kwargs
):
    core, apps = clients

    result = operation(namespace=None, **kwargs)

    assert result == {"success": False, "error": "Invalid namespace"}
    core.assert_not_called()
    apps.assert_not_called()


def test_list_deployments_sanitizes_api_failure(clients):
    _, apps = clients
    apps.list_namespaced_deployment.side_effect = ApiException(
        status=500,
        reason="Internal Server Error",
    )

    result = kt.list_deployments()

    assert result == {
        "success": False,
        "error": "Kubernetes API request failed",
        "status_code": 500,
    }


def test_get_pod_logs_sanitizes_api_failure(clients):
    core, _ = clients
    core.read_namespaced_pod_log.side_effect = ApiException(
        status=500,
        reason="Internal Server Error",
    )

    result = kt.get_pod_logs("sample-pod")

    assert result == {
        "success": False,
        "error": "Kubernetes API request failed",
        "status_code": 500,
    }
