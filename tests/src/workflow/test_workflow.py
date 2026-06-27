import unittest
from typing import Optional
from unittest.mock import Mock

from evoagentx.core.base_config import Parameter
from evoagentx.models.base_model import BaseLLM
from evoagentx.models.model_configs import LLMConfig
from evoagentx.workflow.action_graph import ActionGraph
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


class TestWorkFlow(unittest.IsolatedAsyncioTestCase):

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

    async def test_reusing_workflow_resets_environment_between_runs(self):
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
