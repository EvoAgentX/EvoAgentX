import unittest
from evoagentx.models import OpenAILLMConfig, OpenAILLM 
from evoagentx.workflow.workflow_graph import SEMWorkFlowGraph 
from evoagentx.optimizers.sem_optimizer import SEMWorkFlowScheme



class TestModule(unittest.TestCase):

    def setUp(self):
        self.model = OpenAILLM(config=OpenAILLMConfig(model="gpt-4o-mini", openai_key="XXX"))
        self.graph = SEMWorkFlowGraph(llm=self.model)
        self.scheme = SEMWorkFlowScheme(self.graph)

    def test_python_scheme(self):

        repr = self.scheme.convert_to_scheme(scheme="python")
        new_graph = self.scheme.parse_workflow_python_repr("```python\n" + repr + "\n```")
        self.assertEqual(len(new_graph.nodes), len(self.graph.nodes))
        self.assertEqual(len(new_graph.edges), len(self.graph.edges))
        self.assertFalse(new_graph == self.graph)

        # test empty repr 
        new_graph = self.scheme.parse_workflow_python_repr("")
        self.assertEqual(new_graph, self.graph)

        # test invalid repr 
        new_graph = self.scheme.parse_workflow_python_repr("invalid repr")
        self.assertEqual(new_graph, self.graph)

        # test create new graph  
        steps = eval(repr.replace("steps = ", "").strip())
        new_steps = steps[:4] + [{"name": "test", "args": ["test_input", "code"], "outputs": ["test_output"]}] + steps[4:]
        new_repr = "steps = " + str(new_steps)
        new_graph = self.scheme.parse_workflow_python_repr("```python\n" + new_repr + "\n```")
        new_graph_info = new_graph.get_graph_info() 
        self.assertEqual(len(new_graph_info["tasks"]), 6) 
        self.assertEqual(new_graph_info["tasks"][-2]["name"], "test") 
        new_task_inputs = [input_info["name"] for input_info in new_graph_info["tasks"][-2]["inputs"]]
        self.assertEqual(new_task_inputs, ["test_input", "code"])
        new_task_outputs = [output_info["name"] for output_info in new_graph_info["tasks"][-2]["outputs"]]
        self.assertEqual(new_task_outputs, ["test_output"])
        self.assertFalse(new_graph == self.graph)
    
    def test_yaml_scheme(self):
        repr = self.scheme.convert_to_scheme(scheme="yaml")
        new_graph = self.scheme.parse_workflow_yaml_repr("```yaml\n" + repr + "\n```")
        self.assertEqual(len(new_graph.nodes), len(self.graph.nodes))
        self.assertEqual(len(new_graph.edges), len(self.graph.edges))
        self.assertFalse(new_graph == self.graph)

        # test empty repr 
        new_graph = self.scheme.parse_workflow_yaml_repr("")
        self.assertEqual(new_graph, self.graph)

        # test invalid repr 
        new_graph = self.scheme.parse_workflow_yaml_repr("invalid repr")
        self.assertEqual(new_graph, self.graph)

    def test_code_scheme(self):
        repr = self.scheme.convert_to_scheme(scheme="code")
        new_graph = self.scheme.parse_workflow_code_repr("```code\n" + repr + "\n```")
        self.assertEqual(len(new_graph.nodes), len(self.graph.nodes))
        self.assertEqual(len(new_graph.edges), len(self.graph.edges))
        self.assertFalse(new_graph == self.graph)

        # test empty repr 
        new_graph = self.scheme.parse_workflow_code_repr("")
        self.assertEqual(new_graph, self.graph)

        # test invalid repr 
        new_graph = self.scheme.parse_workflow_code_repr("invalid repr")
        self.assertEqual(new_graph, self.graph)

    def test_bpmn_scheme(self):
        repr = self.scheme.convert_to_scheme(scheme="bpmn")
        new_graph = self.scheme.parse_workflow_bpmn_repr("```bpmn\n" + repr + "\n```")
        self.assertEqual(len(new_graph.nodes), len(self.graph.nodes))
        self.assertEqual(len(new_graph.edges), len(self.graph.edges))
        self.assertFalse(new_graph == self.graph)

        # test empty repr 
        new_graph = self.scheme.parse_workflow_bpmn_repr("")
        self.assertEqual(new_graph, self.graph)

        # test invalid repr 
        new_graph = self.scheme.parse_workflow_bpmn_repr("invalid repr")
        self.assertEqual(new_graph, self.graph)

    def test_core_scheme(self):
        repr = self.scheme.convert_to_scheme(scheme="core")
        new_graph = self.scheme.parse_workflow_core_repr("```core\n" + repr + "\n```")
        self.assertEqual(len(new_graph.nodes), len(self.graph.nodes))
        self.assertEqual(len(new_graph.edges), len(self.graph.edges))
        self.assertFalse(new_graph == self.graph)   
        
        # test empty repr 
        new_graph = self.scheme.parse_workflow_core_repr("")
        self.assertEqual(new_graph, self.graph)

        # test invalid repr 
        new_graph = self.scheme.parse_workflow_core_repr("invalid repr")
        self.assertEqual(new_graph, self.graph)


if __name__ == "__main__":
    unittest.main()