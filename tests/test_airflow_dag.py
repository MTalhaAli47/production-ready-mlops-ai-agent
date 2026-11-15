"""Orchestration tests for the Airflow training DAG."""

import unittest
from datetime import timedelta

try:
    from airflow.models import DagBag
except ModuleNotFoundError:
    DagBag = None


@unittest.skipIf(DagBag is None, "Airflow is tested in its Docker runtime")
class TestAirflowTrainingDAG(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bag = DagBag(
            dag_folder="/opt/airflow/dags",
            safe_mode=False,
            collect_dags=True,
            parse_lang_sdk_files=True,
        )
        cls.dag = cls.bag.dags.get("mlops_a03_training_pipeline")

    def test_dag_loads_without_import_errors(self):
        self.assertEqual(self.bag.import_errors, {})
        self.assertIsNotNone(self.dag)

    def test_task_dependencies(self):
        expected = {
            "validate_stage": set(),
            "preprocess_stage": {"validate_stage"},
            "train_stage": {"preprocess_stage"},
            "evaluate_stage": {
                "validate_stage",
                "preprocess_stage",
                "train_stage",
            },
            "quality_gate_stage": {"evaluate_stage"},
            "promotion_eligibility_stage": {"quality_gate_stage"},
        }

        for task_id, upstream_ids in expected.items():
            with self.subTest(task=task_id):
                task = self.dag.get_task(task_id)
                actual = {item.task_id for item in task.upstream_list}
                self.assertEqual(actual, upstream_ids)

    def test_retry_configuration(self):
        expected = {
            "validate_stage": 2,
            "preprocess_stage": 2,
            "train_stage": 2,
            "evaluate_stage": 1,
            "quality_gate_stage": 0,
            "promotion_eligibility_stage": 0,
        }

        for task_id, retries in expected.items():
            with self.subTest(task=task_id):
                task = self.dag.get_task(task_id)
                self.assertEqual(task.retries, retries)

        for task_id in (
            "validate_stage",
            "preprocess_stage",
            "train_stage",
            "evaluate_stage",
        ):
            with self.subTest(retry_delay=task_id):
                self.assertEqual(
                    self.dag.get_task(task_id).retry_delay,
                    timedelta(minutes=1),
                )

    def test_gate_failure_blocks_promotion(self):
        gate = self.dag.get_task("quality_gate_stage")
        promotion = self.dag.get_task("promotion_eligibility_stage")

        self.assertEqual(gate.trigger_rule, "all_success")
        self.assertEqual(promotion.trigger_rule, "all_success")
        self.assertIn(gate.task_id, {
            task.task_id for task in promotion.upstream_list
        })

    def test_promotion_callable_rejects_ineligible_model(self):
        promotion = self.dag.get_task("promotion_eligibility_stage")

        with self.assertRaisesRegex(
            ValueError,
            "Model is not eligible for promotion",
        ):
            promotion.python_callable({"eligible": False})

    def test_promotion_callable_is_decision_only(self):
        promotion = self.dag.get_task("promotion_eligibility_stage")

        result = promotion.python_callable({
            "eligible": True,
            "dataset_rows": 30,
        })

        self.assertEqual(result["promotion_eligible"], True)
        self.assertEqual(
            result["action"],
            "decision_only_no_registry_or_deployment",
        )


if __name__ == "__main__":
    unittest.main()
