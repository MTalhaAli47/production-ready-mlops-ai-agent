"""Safe, read-only Kubernetes inspection tools for the MLOps A03 project."""

import re

from kubernetes.client.exceptions import ApiException

from kubernetes import client, config

DEFAULT_NAMESPACE = "mlops-a03"
APPROVED_NAMESPACES = frozenset({DEFAULT_NAMESPACE})

MAX_RESULTS = 100
MAX_LOG_LINES = 1000
MAX_LOG_BYTES = 1024 * 1024
MAX_CONTINUE_TOKEN_LENGTH = 4096

RESOURCE_NAME_PATTERN = re.compile(
    r"^[a-z0-9](?:[-a-z0-9.]*[a-z0-9])?$"
)
CONTAINER_NAME_PATTERN = re.compile(r"^[a-z0-9](?:[-a-z0-9_.-]*[a-z0-9])?$")


def _get_clients():
    """Load Kubernetes configuration and create API clients."""
    try:
        config.load_kube_config()
    except config.ConfigException:
        try:
            config.load_incluster_config()
        except config.ConfigException as exc:
            raise RuntimeError(
                "Kubernetes configuration is unavailable"
            ) from exc

    return client.CoreV1Api(), client.AppsV1Api()


def _validate_namespace(namespace):
    """Allow access only to explicitly approved namespaces."""
    if (
        not isinstance(namespace, str)
        or not RESOURCE_NAME_PATTERN.fullmatch(namespace)
    ):
        return {"success": False, "error": "Invalid namespace"}

    if namespace not in APPROVED_NAMESPACES:
        return {"success": False, "error": "Namespace is not approved"}

    return None


def _validate_name(name, label="resource name"):
    """Validate a Kubernetes resource name."""
    if (
        not isinstance(name, str)
        or len(name) > 253
        or not RESOURCE_NAME_PATTERN.fullmatch(name)
    ):
        return {"success": False, "error": f"Invalid {label}"}

    return None


def _validate_pagination(limit, continue_token):
    """Validate bounded Kubernetes list pagination."""
    if (
        not isinstance(limit, int)
        or isinstance(limit, bool)
        or not 1 <= limit <= MAX_RESULTS
    ):
        return {"success": False, "error": "Invalid limit"}

    if continue_token is not None and (
        not isinstance(continue_token, str)
        or not continue_token
        or len(continue_token) > MAX_CONTINUE_TOKEN_LENGTH
        or any(ord(character) < 32 for character in continue_token)
    ):
        return {"success": False, "error": "Invalid continuation token"}

    return None


def _failure(exc):
    """Convert client and API exceptions into safe structured errors."""
    if isinstance(exc, ApiException):
        status_code = exc.status
        if status_code == 404:
            return {
                "success": False,
                "error": "Kubernetes resource was not found",
                "status_code": status_code,
            }
        if status_code == 403:
            return {
                "success": False,
                "error": "Kubernetes API access is forbidden",
                "status_code": status_code,
            }
        return {
            "success": False,
            "error": "Kubernetes API request failed",
            "status_code": status_code,
        }

    if isinstance(exc, RuntimeError):
        return {"success": False, "error": str(exc)}

    return {
        "success": False,
        "error": f"Kubernetes client failed: {type(exc).__name__}",
    }


def _isoformat(value):
    return value.isoformat() if value is not None else None


def _metadata(resource):
    metadata = resource.metadata
    return {
        "name": metadata.name,
        "namespace": metadata.namespace,
        "uid": metadata.uid,
        "labels": metadata.labels or {},
        "creation_timestamp": _isoformat(metadata.creation_timestamp),
    }


def _condition_summary(conditions):
    return [
        {
            "type": item.type,
            "status": item.status,
            "reason": item.reason,
            "message": item.message,
            "last_transition_time": _isoformat(
                item.last_transition_time
            ),
        }
        for item in (conditions or [])
    ]


def _pod_summary(pod):
    status = pod.status
    container_statuses = []

    for item in status.container_statuses or []:
        state = item.state
        waiting = state.waiting
        terminated = state.terminated
        running = state.running

        container_statuses.append(
            {
                "name": item.name,
                "ready": item.ready,
                "restart_count": item.restart_count,
                "image": item.image,
                "image_id": item.image_id,
                "state": (
                    "waiting"
                    if waiting
                    else "terminated"
                    if terminated
                    else "running"
                    if running
                    else "unknown"
                ),
                "reason": (
                    waiting.reason
                    if waiting
                    else terminated.reason
                    if terminated
                    else None
                ),
                "message": (
                    waiting.message
                    if waiting
                    else terminated.message
                    if terminated
                    else None
                ),
                "exit_code": terminated.exit_code if terminated else None,
            }
        )

    return {
        **_metadata(pod),
        "phase": status.phase,
        "pod_ip": status.pod_ip,
        "node_name": pod.spec.node_name,
        "ready": any(
            item.type == "Ready" and item.status == "True"
            for item in (status.conditions or [])
        ),
        "conditions": _condition_summary(status.conditions),
        "container_statuses": container_statuses,
    }


def _deployment_summary(deployment):
    status = deployment.status
    spec = deployment.spec

    return {
        **_metadata(deployment),
        "desired_replicas": spec.replicas or 0,
        "ready_replicas": status.ready_replicas or 0,
        "available_replicas": status.available_replicas or 0,
        "updated_replicas": status.updated_replicas or 0,
        "unavailable_replicas": status.unavailable_replicas or 0,
        "generation": deployment.metadata.generation,
        "observed_generation": status.observed_generation,
        "conditions": _condition_summary(status.conditions),
        "selector": spec.selector.match_labels or {},
        "images": [
            item.image for item in (spec.template.spec.containers or [])
        ],
    }


def _service_summary(service):
    spec = service.spec

    return {
        **_metadata(service),
        "type": spec.type,
        "cluster_ip": spec.cluster_ip,
        "external_ips": spec.external_ips or [],
        "selector": spec.selector or {},
        "ports": [
            {
                "name": item.name,
                "protocol": item.protocol,
                "port": item.port,
                "target_port": str(item.target_port),
                "node_port": item.node_port,
            }
            for item in (spec.ports or [])
        ],
    }


def _list_core_resources(
    method_name,
    namespace,
    limit,
    continue_token,
    serializer,
):
    namespace_error = _validate_namespace(namespace)
    if namespace_error:
        return namespace_error

    pagination_error = _validate_pagination(limit, continue_token)
    if pagination_error:
        return pagination_error

    try:
        core_api, _ = _get_clients()
        response = getattr(core_api, method_name)(
            namespace=namespace,
            limit=limit,
            _continue=continue_token,
        )
        items = [serializer(item) for item in response.items]

        return {
            "success": True,
            "data": items,
            "count": len(items),
            "continue_token": response.metadata._continue or None,
        }
    except (ApiException, RuntimeError, config.ConfigException, OSError) as exc:
        return _failure(exc)


def list_pods(namespace=DEFAULT_NAMESPACE, limit=50, continue_token=None):
    """List pods in the approved namespace."""
    return _list_core_resources(
        "list_namespaced_pod",
        namespace,
        limit,
        continue_token,
        _pod_summary,
    )


def get_pod_details(name, namespace=DEFAULT_NAMESPACE):
    """Retrieve details for one pod in the approved namespace."""
    namespace_error = _validate_namespace(namespace)
    if namespace_error:
        return namespace_error

    name_error = _validate_name(name, "pod name")
    if name_error:
        return name_error

    try:
        core_api, _ = _get_clients()
        pod = core_api.read_namespaced_pod(name=name, namespace=namespace)
        return {"success": True, "data": _pod_summary(pod)}
    except (ApiException, RuntimeError, config.ConfigException, OSError) as exc:
        return _failure(exc)


def list_deployments(
    namespace=DEFAULT_NAMESPACE,
    limit=50,
    continue_token=None,
):
    """List deployments in the approved namespace."""
    namespace_error = _validate_namespace(namespace)
    if namespace_error:
        return namespace_error

    pagination_error = _validate_pagination(limit, continue_token)
    if pagination_error:
        return pagination_error

    try:
        _, apps_api = _get_clients()
        response = apps_api.list_namespaced_deployment(
            namespace=namespace,
            limit=limit,
            _continue=continue_token,
        )
        items = [_deployment_summary(item) for item in response.items]

        return {
            "success": True,
            "data": items,
            "count": len(items),
            "continue_token": response.metadata._continue or None,
        }
    except (ApiException, RuntimeError, config.ConfigException, OSError) as exc:
        return _failure(exc)


def get_deployment_details(name, namespace=DEFAULT_NAMESPACE):
    """Retrieve deployment rollout and replica details."""
    namespace_error = _validate_namespace(namespace)
    if namespace_error:
        return namespace_error

    name_error = _validate_name(name, "deployment name")
    if name_error:
        return name_error

    try:
        _, apps_api = _get_clients()
        deployment = apps_api.read_namespaced_deployment(
            name=name,
            namespace=namespace,
        )
        return {"success": True, "data": _deployment_summary(deployment)}
    except (ApiException, RuntimeError, config.ConfigException, OSError) as exc:
        return _failure(exc)


def list_services(
    namespace=DEFAULT_NAMESPACE,
    limit=50,
    continue_token=None,
):
    """List Services in the approved namespace."""
    return _list_core_resources(
        "list_namespaced_service",
        namespace,
        limit,
        continue_token,
        _service_summary,
    )


def get_service_details(name, namespace=DEFAULT_NAMESPACE):
    """Retrieve one Service's type, ports, and selector."""
    namespace_error = _validate_namespace(namespace)
    if namespace_error:
        return namespace_error

    name_error = _validate_name(name, "service name")
    if name_error:
        return name_error

    try:
        core_api, _ = _get_clients()
        service = core_api.read_namespaced_service(
            name=name,
            namespace=namespace,
        )
        return {"success": True, "data": _service_summary(service)}
    except (ApiException, RuntimeError, config.ConfigException, OSError) as exc:
        return _failure(exc)


def get_pod_logs(
    name,
    namespace=DEFAULT_NAMESPACE,
    container=None,
    tail_lines=200,
    timestamps=False,
):
    """Retrieve a bounded, non-following log snapshot from a pod."""
    namespace_error = _validate_namespace(namespace)
    if namespace_error:
        return namespace_error

    name_error = _validate_name(name, "pod name")
    if name_error:
        return name_error

    if container is not None and (
        not isinstance(container, str)
        or not CONTAINER_NAME_PATTERN.fullmatch(container)
    ):
        return {"success": False, "error": "Invalid container name"}

    if (
        not isinstance(tail_lines, int)
        or isinstance(tail_lines, bool)
        or not 1 <= tail_lines <= MAX_LOG_LINES
    ):
        return {"success": False, "error": "Invalid tail_lines"}

    if not isinstance(timestamps, bool):
        return {"success": False, "error": "Invalid timestamps"}

    try:
        core_api, _ = _get_clients()
        logs = core_api.read_namespaced_pod_log(
            name=name,
            namespace=namespace,
            container=container,
            tail_lines=tail_lines,
            timestamps=timestamps,
            limit_bytes=MAX_LOG_BYTES,
            follow=False,
        )
        return {
            "success": True,
            "data": {
                "pod": name,
                "namespace": namespace,
                "container": container,
                "logs": logs,
                "tail_lines": tail_lines,
                "max_bytes": MAX_LOG_BYTES,
            },
        }
    except (ApiException, RuntimeError, config.ConfigException, OSError) as exc:
        return _failure(exc)


def list_events(
    namespace=DEFAULT_NAMESPACE,
    limit=50,
    continue_token=None,
):
    """List Kubernetes events in the approved namespace."""
    namespace_error = _validate_namespace(namespace)
    if namespace_error:
        return namespace_error

    pagination_error = _validate_pagination(limit, continue_token)
    if pagination_error:
        return pagination_error

    try:
        core_api, _ = _get_clients()
        response = core_api.list_namespaced_event(
            namespace=namespace,
            limit=limit,
            _continue=continue_token,
        )

        items = [
            {
                **_metadata(event),
                "type": event.type,
                "reason": event.reason,
                "message": event.message,
                "count": event.count,
                "first_timestamp": _isoformat(event.first_timestamp),
                "last_timestamp": _isoformat(event.last_timestamp),
                "involved_object": {
                    "kind": event.involved_object.kind,
                    "name": event.involved_object.name,
                    "namespace": event.involved_object.namespace,
                },
            }
            for event in response.items
        ]

        return {
            "success": True,
            "data": items,
            "count": len(items),
            "continue_token": response.metadata._continue or None,
        }
    except (ApiException, RuntimeError, config.ConfigException, OSError) as exc:
        return _failure(exc)


def diagnose_pod(name, namespace=DEFAULT_NAMESPACE):
    """Summarize pod health using its phase, readiness, and container states."""
    result = get_pod_details(name=name, namespace=namespace)
    if not result["success"]:
        return result

    pod = result["data"]
    issues = []

    if pod["phase"] == "Failed":
        issues.append("Pod phase is Failed")
    elif pod["phase"] == "Pending":
        issues.append("Pod is Pending")

    for item in pod["container_statuses"]:
        if item["reason"]:
            issues.append(f"Container {item['name']}: {item['reason']}")

        if item["state"] == "terminated" and item["exit_code"] not in (None, 0):
            issues.append(
                f"Container {item['name']} exited with code {item['exit_code']}"
            )

    if not pod["ready"] and not issues:
        issues.append("Pod is not Ready")

    if pod["phase"] == "Running" and pod["ready"] and not issues:
        health = "healthy"
        summary = "Pod is running and ready"
    elif issues:
        health = "unhealthy"
        summary = "; ".join(issues)
    else:
        health = "unknown"
        summary = f"Pod phase is {pod['phase'] or 'unknown'}"

    return {
        "success": True,
        "data": {
            "name": pod["name"],
            "namespace": pod["namespace"],
            "phase": pod["phase"],
            "ready": pod["ready"],
            "health": health,
            "summary": summary,
            "issues": issues,
            "container_statuses": pod["container_statuses"],
            "conditions": pod["conditions"],
        },
    }
