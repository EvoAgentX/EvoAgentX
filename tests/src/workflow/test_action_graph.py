import unittest
import os
from unittest.mock import patch
from evoagentx.models import OpenAILLMConfig 
from evoagentx.workflow.action_graph import ActionGraph, QAActionGraph


class TestModule(unittest.TestCase):

    def setUp(self):
        self.llm_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key="XXX")
        self.qa_action_graph = QAActionGraph(llm_config=self.llm_config, name="QAActionGraph", description="This workflow aims to address multi-hop QA tasks.") 
    
    @patch('evoagentx.workflow.operators.AnswerGenerate.execute')
    @patch('evoagentx.workflow.operators.ScEnsemble.execute')
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
    
    def test_get_graph_info(self):
        graph_info = self.qa_action_graph.get_graph_info()
        self.assertEqual(graph_info["name"], "QAActionGraph")
        self.assertEqual(graph_info["description"], "This workflow aims to address multi-hop QA tasks.")
        self.assertEqual(len(graph_info["operators"]), 2)
        self.assertEqual(graph_info["operators"]["answer_generate"]["name"], "AnswerGenerate")
        self.assertEqual(graph_info["operators"]["sc_ensemble"]["name"], "ScEnsemble")
    
    def test_from_dict(self):
        graph_info = self.qa_action_graph.get_graph_info()
        graph_info["operators"]["answer_generate"]["prompt"] = "This is a mocked prompt"
        loaded_graph = ActionGraph.from_dict(graph_info)
        self.assertEqual(loaded_graph.name, "QAActionGraph")
        self.assertEqual(loaded_graph.description, "This workflow aims to address multi-hop QA tasks.")
        self.assertEqual(loaded_graph.answer_generate.name, "AnswerGenerate")
        self.assertEqual(loaded_graph.answer_generate.prompt, "This is a mocked prompt")
        self.assertEqual(loaded_graph.sc_ensemble.name, "ScEnsemble")

    def test_save_and_load(self):
        self.qa_action_graph.save_module("tests/src/workflow/saved_qa_action_graph.json")
        loaded_graph = ActionGraph.from_file("tests/src/workflow/saved_qa_action_graph.json")
        self.assertEqual(loaded_graph.name, "QAActionGraph")
        self.assertEqual(loaded_graph.description, "This workflow aims to address multi-hop QA tasks.")
        self.assertEqual(loaded_graph.answer_generate.name, "AnswerGenerate")
        self.assertEqual(loaded_graph.sc_ensemble.name, "ScEnsemble")

    def tearDown(self):
        if os.path.exists("tests/src/workflow/saved_qa_action_graph.json"):
            os.remove("tests/src/workflow/saved_qa_action_graph.json")
