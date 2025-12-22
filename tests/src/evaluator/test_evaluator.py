import unittest
from unittest.mock import Mock, patch
from evoagentx.evaluators.evaluator import Evaluator
from evoagentx.benchmark.benchmark import Benchmark
from evoagentx.workflow.workflow_graph import WorkFlowGraph
from evoagentx.workflow.action_graph import ActionGraph
from evoagentx.agents.agent_manager import AgentManager
from evoagentx.models.base_model import BaseLLM

class TestEvaluator(unittest.TestCase):

    def setUp(self):
        # Create mock objects
        self.benchmark = Mock(spec=Benchmark)
        self.benchmark.get_test_data.return_value = [
            {"id": "1", "input": "test1"},
            {"id": "2", "input": "test2"}
        ]
        self.benchmark.get_id.side_effect = lambda example: example["id"]
        self.benchmark.get_label.return_value = "expected"
        self.benchmark.evaluate.return_value = {"accuracy": 1.0}

        # Mock LLM
        self.llm = Mock(spec=BaseLLM)
        
        # Mock agent manager
        self.agent_manager = Mock(spec=AgentManager)

        # Create mock graphs
        self.workflow_graph = Mock(spec=WorkFlowGraph)
        self.action_graph = Mock(spec=ActionGraph)
        self.action_graph.execute.return_value = {"output": "prediction"}

        # Create evaluator instance
        self.evaluator = Evaluator(
            llm=self.llm,
            num_workers=1,
            agent_manager=self.agent_manager
        )
    
    @patch.object(Evaluator, '_execute_workflow_graph')
    def test_single_thread_evaluation_workflow_graph(self, mock_execute):
        # Set up mock return value
        mock_execute.return_value = ("workflow_graph_prediction", ["trajectory_data"])
        
        # Test evaluation with single thread using WorkFlowGraph
        results = self.evaluator.evaluate(
            graph=self.workflow_graph,
            benchmark=self.benchmark,
            eval_mode="test"
        )

        # Verify results
        self.assertEqual(mock_execute.call_count, 2) 
        self.assertEqual(results, {"accuracy": 1.0})
        self.assertEqual(len(self.evaluator.get_all_evaluation_records()), 2)

    def test_single_thread_evaluation_action_graph(self):
        # Test evaluation with single thread using ActionGraph
        results = self.evaluator.evaluate(
            graph=self.action_graph,
            benchmark=self.benchmark,
            eval_mode="test"
        )

        # Verify results
        self.assertEqual(results, {"accuracy": 1.0})
        self.assertEqual(len(self.evaluator.get_all_evaluation_records()), 2)

    # @patch.object(Evaluator, '_execute_workflow_graph')
    # def test_multi_thread_evaluation(self, mock_execute):
    #     # Set up mock return value for workflow graph
    #     mock_execute.return_value = ("workflow_graph_prediction", ["trajectory_data"])
        
    #     # Test evaluation with multiple threads
    #     evaluator = Evaluator(
    #         llm=self.llm,
    #         num_workers=2,
    #         agent_manager=self.agent_manager,
    #         verbose=True
    #     )

    #     # Test workflow graph
    #     results_workflow = evaluator.evaluate(
    #         graph=self.workflow_graph,
    #         benchmark=self.benchmark,
    #         eval_mode="test"
    #     )
    #     self.assertEqual(results_workflow, {"accuracy": 1.0})
    #     self.assertEqual(len(evaluator.get_all_evaluation_records()), 2)
    #     self.assertEqual(mock_execute.call_count, 2)

    #     # Clear evaluation records for next test
    #     evaluator._evaluation_records.clear()

    #     # Test action graph
    #     results_action = evaluator.evaluate(
    #         graph=self.action_graph,
    #         benchmark=self.benchmark,
    #         eval_mode="test"
    #     )
    #     self.assertEqual(results_action, {"accuracy": 1.0})
    #     self.assertEqual(len(evaluator.get_all_evaluation_records()), 2)

    #     # Verify that records contain the expected data
    #     records = evaluator.get_all_evaluation_records()
    #     for record in records.values():
    #         self.assertIn("prediction", record)
    #         self.assertIn("label", record)
    #         self.assertIn("metrics", record)
    #         self.assertEqual(record["label"], "expected")
    #         self.assertEqual(record["metrics"], {"accuracy": 1.0})

    def test_evaluation_with_custom_collate(self):
        # Test evaluation with custom collate function
        def collate_func(x):
            return {"processed_" + k: v for k, v in x.items()}

        evaluator = Evaluator(
            llm=self.llm,
            num_workers=1,
            collate_func=collate_func
        )
        evaluator.evaluate(
            graph=self.action_graph,
            benchmark=self.benchmark,
            eval_mode="test"
        )

        # Get the first call's arguments
        call_args = self.action_graph.execute.call_args_list[0][1]
        self.assertTrue(all(k.startswith("processed_") for k in call_args.keys()))
    
    def test_evaluation_with_output_postprocess(self):
        # Test evaluation with output postprocess function
        def postprocess_func(x):
            return x["output"].upper()

        evaluator = Evaluator(
            llm=self.llm,
            num_workers=1,
            output_postprocess_func=postprocess_func
        )
        
        evaluator.evaluate(
            graph=self.action_graph,
            benchmark=self.benchmark,
            eval_mode="test"
        )

        records = evaluator.get_all_evaluation_records()
        for record in records.values():
            self.assertEqual(record["prediction"], "PREDICTION")
    
    def test_get_example_evaluation_record(self):
        # Test get example evaluation record
        self.evaluator.evaluate(
            graph=self.action_graph,
            benchmark=self.benchmark,
            eval_mode="test"
        )

        example = {"id": "1", "input": "test1"}
        record = self.evaluator.get_example_evaluation_record(self.benchmark, example)

        self.assertIsNotNone(record)
        self.assertEqual(record["prediction"], {"output": "prediction"})
        self.assertEqual(record["label"], "expected")
        self.assertEqual(record["metrics"], {"accuracy": 1.0})
    
    def test_invalid_eval_mode(self):
        with self.assertRaises(AssertionError):
            self.evaluator.evaluate(
                graph=self.action_graph,
                benchmark=self.benchmark,
                eval_mode="invalid"
            )

    def test_empty_data_evaluation(self):
        # Test empty data evaluation
        self.benchmark.get_test_data.return_value = []
        results = self.evaluator.evaluate(
            graph=self.action_graph,
            benchmark=self.benchmark,
            eval_mode="test"
        )
        
        self.assertEqual(results, {})
        self.assertEqual(len(self.evaluator.get_all_evaluation_records()), 0)

if __name__ == '__main__':
    unittest.main()