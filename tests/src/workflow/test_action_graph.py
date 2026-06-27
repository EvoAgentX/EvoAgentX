import unittest
import os
from typing import Optional
from unittest.mock import Mock, patch

from evoagentx.core.base_config import Parameter
from evoagentx.models import OpenAILLMConfig 
from evoagentx.models.base_model import BaseLLM
from evoagentx.models.model_configs import LLMConfig
from evoagentx.workflow.action_graph import ActionGraph, QAActionGraph
from evoagentx.workflow.workflow import WorkFlow
from evoagentx.workflow.workflow_graph import WorkFlowGraph, WorkFlowNode


class EchoActionGraph(ActionGraph):
    llm_config: Optional[LLMConfig] = None

    def init_module(self):
        pass

    def execute(self, value: str) -> dict:
        return {"result": value}

    async def async_execute(self, value: str) -> dict:
        return {"result": value}


class TestModule(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.llm_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key="XXX")
        self.qa_action_graph = QAActionGraph(llm_config=self.llm_config, name="QAActionGraph", description="This workflow aims to address multi-hop QA tasks.") 
    
    @patch('evoagentx.workflow.operators.AnswerGenerate.execute')
    @patch('evoagentx.workflow.operators.QAScEnsemble.execute')
    def test_execute(self, mock_sc_ensemble, mock_answer_generate):
        """Test execute method with mocked operators"""
        # Set up mock return values
        mock_answer_generate.return_value = {"answer": "This is a mocked answer"}
        mock_sc_ensemble.return_value = {"response": "final answer"}
        
        # Test execution
        result = self.qa_action_graph.execute(problem="This is a test problem.")
        
        # Verify the operators were called with correct arguments
        self.assertTrue(mock_answer_generate.called)
        self.assertTrue(mock_sc_ensemble.called)
        
        # Verify the result contains both answer and score
        self.assertEqual(result["answer"], "final answer")
    
    @patch('evoagentx.workflow.operators.AnswerGenerate.async_execute')
    @patch('evoagentx.workflow.operators.QAScEnsemble.async_execute')
    async def test_async_execute(self, mock_sc_ensemble, mock_answer_generate):
        """Test async_execute method with mocked async operators"""
        # Set up mock return values
        mock_answer_generate.return_value = {"answer": "This is a mocked async answer"}
        mock_sc_ensemble.return_value = {"response": "final async answer"}
        
        # Test async execution
        result = await self.qa_action_graph.async_execute(problem="This is a test async problem.")
        
        # Verify the operators were called with correct arguments
        self.assertTrue(mock_answer_generate.called)
        self.assertTrue(mock_sc_ensemble.called)
        
        # Verify the result contains correct answer
        self.assertEqual(result["answer"], "final async answer")
    
    def test_get_graph_info(self):
        graph_info = self.qa_action_graph.get_graph_info()
        self.assertEqual(graph_info["name"], "QAActionGraph")
        self.assertEqual(graph_info["description"], "This workflow aims to address multi-hop QA tasks.")
        self.assertEqual(len(graph_info["operators"]), 2)
        self.assertEqual(graph_info["operators"]["answer_generate"]["name"], "AnswerGenerate")
        self.assertEqual(graph_info["operators"]["sc_ensemble"]["name"], "QAScEnsemble")
    
    def test_from_dict(self):
        graph_info = self.qa_action_graph.get_graph_info()
        graph_info["operators"]["answer_generate"]["prompt"] = "This is a mocked prompt"
        graph_info["llm_config"] = self.llm_config.to_dict()
        loaded_graph = ActionGraph.from_dict(graph_info)
        self.assertEqual(loaded_graph.name, "QAActionGraph")
        self.assertEqual(loaded_graph.description, "This workflow aims to address multi-hop QA tasks.")
        self.assertEqual(loaded_graph.answer_generate.name, "AnswerGenerate")
        self.assertEqual(loaded_graph.answer_generate.prompt, "This is a mocked prompt")
        self.assertEqual(loaded_graph.sc_ensemble.name, "QAScEnsemble")

    def test_save_and_load(self):
        self.qa_action_graph.save_module("tests/src/workflow/saved_qa_action_graph.json")
        loaded_graph = ActionGraph.from_file("tests/src/workflow/saved_qa_action_graph.json", llm_config=self.llm_config)
        self.assertEqual(loaded_graph.name, "QAActionGraph")
        self.assertEqual(loaded_graph.description, "This workflow aims to address multi-hop QA tasks.")
        self.assertEqual(loaded_graph.answer_generate.name, "AnswerGenerate")
        self.assertEqual(loaded_graph.sc_ensemble.name, "QAScEnsemble")

    def tearDown(self):
        if os.path.exists("tests/src/workflow/saved_qa_action_graph.json"):
            os.remove("tests/src/workflow/saved_qa_action_graph.json")


class TestActionGraphWorkflow(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.workflow = WorkFlow(
            graph=WorkFlowGraph(
                goal="Echo input value",
                nodes=[
                    WorkFlowNode(
                        name="EchoTask",
                        description="Echo the provided input value.",
                        inputs=[Parameter(name="value", type="string", description="Input value")],
                        outputs=[Parameter(name="result", type="string", description="Echo result")],
                        action_graph=EchoActionGraph(
                            name="EchoActionGraph",
                            description="Echoes the input value.",
                        ),
                    )
                ],
            ),
            llm=Mock(spec=BaseLLM),
        )

    async def test_reusing_workflow_resets_environment_between_action_graph_runs(self):
        first_result = await self.workflow.async_execute(inputs={"value": "first"})

        self.assertEqual(first_result.status, "success")
        self.assertEqual(first_result.result, {"result": "first"})
        self.assertGreater(len(self.workflow.environment.trajectory), 0)
        self.assertEqual(self.workflow.environment.execution_data["value"], "first")

        self.workflow.environment.execution_data["stale"] = "leaked"
        self.workflow.environment.task_execution_history.append("StaleTask")

        second_result = await self.workflow.async_execute(inputs={"value": "second"})

        self.assertEqual(second_result.status, "success")
        self.assertEqual(second_result.result, {"result": "second"})
        self.assertEqual(self.workflow.environment.execution_data["value"], "second")
        self.assertEqual(self.workflow.environment.execution_data["result"], "second")
        self.assertNotIn("stale", self.workflow.environment.execution_data)
        self.assertNotIn("StaleTask", self.workflow.environment.task_execution_history)
