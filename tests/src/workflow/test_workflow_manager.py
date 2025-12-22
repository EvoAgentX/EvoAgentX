import unittest
import pytest
from unittest.mock import Mock, patch, AsyncMock

from evoagentx.core.message import Message, MessageType
from evoagentx.core.base_config import Parameter
from evoagentx.models.base_model import BaseLLM, LLMOutputParser
from evoagentx.workflow.environment import Environment, TrajectoryState
from evoagentx.workflow.workflow_graph import WorkFlowNode, WorkFlowGraph, WorkFlowEdge, WorkFlowNodeState
from evoagentx.workflow.workflow_manager import TaskSchedulerOutput, NextAction, WorkFlowManager


class MockLLMOutputParser(LLMOutputParser):
    content: str = "Test output"
    
    def to_str(self, **kwargs) -> str:
        return self.content


class TestWorkFlowManager(unittest.TestCase):

    def setUp(self):
        # Create mock LLM
        self.mock_llm = Mock(spec=BaseLLM)
        self.mock_llm.generate = Mock()
        self.mock_llm.async_generate = AsyncMock()

        # Set up output parsers for mock responses
        self.task_output = TaskSchedulerOutput(
            decision="forward",
            task_name="Task2",
            reason="This is the next logical step"
        )
        
        self.action_output = NextAction(
            agent="TestAgent",
            action="TestAction",
            reason="This is the appropriate action"
        )
        
        self.mock_llm.generate.return_value = self.task_output
        self.mock_llm.async_generate.return_value = self.task_output

        # Create a workflow manager
        self.workflow_manager = WorkFlowManager(llm=self.mock_llm)
        
        # Create a workflow graph
        self.create_test_workflow()
        
        # Create an environment
        self.env = Environment()

    def create_test_workflow(self):
        """Create a test workflow with 3 tasks in sequence"""
        # Create 3 nodes
        task1 = WorkFlowNode(
            name="Task1",
            description="First task",
            inputs=[Parameter(name="input1", type="string", description="Input 1")],
            outputs=[Parameter(name="output1", type="string", description="Output 1")],
            agents=["TestAgent"],
            status=WorkFlowNodeState.PENDING
        )
        
        task2 = WorkFlowNode(
            name="Task2",
            description="Second task",
            inputs=[Parameter(name="output1", type="string", description="Output from Task1")],
            outputs=[Parameter(name="output2", type="string", description="Output 2")],
            agents=["TestAgent"],
            status=WorkFlowNodeState.PENDING
        )
        
        task3 = WorkFlowNode(
            name="Task3",
            description="Third task",
            inputs=[Parameter(name="output2", type="string", description="Output from Task2")],
            outputs=[Parameter(name="final_output", type="string", description="Final output")],
            agents=["TestAgent"],
            status=WorkFlowNodeState.PENDING
        )
        
        # Create edges
        edge1 = WorkFlowEdge(source="Task1", target="Task2")
        edge2 = WorkFlowEdge(source="Task2", target="Task3")
        
        # Create the workflow graph
        self.workflow = WorkFlowGraph(
            goal="Test Workflow",
            nodes=[task1, task2, task3],
            edges=[edge1, edge2]
        )

    def test_workflow_initialization(self):
        """Test that the workflow manager is correctly initialized"""
        self.assertIsNotNone(self.workflow_manager)
        self.assertEqual(self.mock_llm, self.workflow_manager.llm)
        self.assertIsNotNone(self.workflow_manager.task_scheduler)
        self.assertIsNotNone(self.workflow_manager.action_scheduler)

    @pytest.mark.asyncio
    @patch('evoagentx.workflow.workflow_manager.TaskScheduler.async_execute')
    async def test_sync_task_scheduling_with_single_task(self, mock_task_scheduler_execute):
        """Test that the task scheduler correctly handles the case of a single candidate task"""
        # Set up mock task scheduler to return a single task
        single_task_output = TaskSchedulerOutput(
            decision="forward",
            task_name="Task2",
            reason="Only one candidate task is available"
        )
        mock_task_scheduler_execute.return_value = single_task_output
        
        # Mark Task1 as completed to make Task2 the only next candidate
        self.workflow.set_node_status("Task1", WorkFlowNodeState.COMPLETED)
        
        # Call schedule_next_task
        task = await self.workflow_manager.schedule_next_task(graph=self.workflow, env=self.env)
        self.assertEqual("Task2", task.name)
        
        # Check if task_scheduler.execute was called
        mock_task_scheduler_execute.assert_called_once()
        
        # Check that the message was added to the environment
        self.assertEqual(1, len(self.env.trajectory))
        message = self.env.trajectory[0].message
        self.assertIsInstance(message.content, TaskSchedulerOutput)
        self.assertEqual("Task2", message.content.task_name)
        self.assertEqual(MessageType.COMMAND, message.msg_type)

    @pytest.mark.asyncio
    @patch('evoagentx.workflow.workflow_manager.ActionScheduler.async_execute')
    async def test_action_scheduling(self, mock_action_scheduler_execute):
        """Test scheduling the next action for a task"""
        # Set up mock action scheduler
        mock_action_scheduler_execute.return_value = (self.action_output, "mock prompt")
        
        # Get the first task node
        task = self.workflow.get_node("Task1")
        
        # Create a mock agent manager
        mock_agent_manager = Mock()
        
        # Call schedule_next_action
        action = await self.workflow_manager.schedule_next_action(
            goal="Test Goal", 
            task=task, 
            agent_manager=mock_agent_manager, 
            env=self.env
        )
        self.assertEqual(self.action_output, action) 
        
        # Check if action_scheduler.execute was called
        mock_action_scheduler_execute.assert_called_once()
        
        # Check that the message was added to the environment
        self.assertEqual(1, len(self.env.trajectory))
        message = self.env.trajectory[0].message
        self.assertIsInstance(message.content, NextAction)
        self.assertEqual("TestAgent", message.content.agent)
        self.assertEqual("TestAction", message.content.action)
        self.assertEqual(MessageType.COMMAND, message.msg_type)

    @pytest.mark.asyncio
    async def test_async_task_scheduling(self):
        """Test async task scheduling with multiple candidate tasks"""
        # Set up the llm.async_generate to return a task
        self.mock_llm.async_generate.return_value = self.task_output
        
        # Run the test
        task = await self.workflow_manager.schedule_next_task(graph=self.workflow, env=self.env)
        
        # Check results
        self.assertIsNotNone(task)
        self.assertEqual("Task2", task.name)
        
        # Verify the environment was updated
        self.assertEqual(1, len(self.env.trajectory))
        message = self.env.trajectory[0].message
        self.assertEqual(self.task_output, message.content)
        self.assertEqual(TrajectoryState.COMPLETED, self.env.trajectory[0].status)

    @pytest.mark.asyncio
    async def test_async_action_scheduling(self):
        """Test async action scheduling"""
        # Set up the llm.async_generate to return an action
        self.mock_llm.async_generate.return_value = self.action_output
        
        # Get the first task node
        task = self.workflow.get_node("Task1")
        
        # Create a mock agent manager
        mock_agent_manager = Mock()
        
        # Run the test
        action = await self.workflow_manager.schedule_next_action(
            goal="Test Goal", 
            task=task, 
            agent_manager=mock_agent_manager, 
            env=self.env
        )
        
        # Check results
        self.assertIsNotNone(action)
        self.assertEqual("TestAgent", action.agent)
        self.assertEqual("TestAction", action.action)
        
        # Verify the environment was updated
        self.assertEqual(1, len(self.env.trajectory))
        message = self.env.trajectory[0].message
        self.assertEqual(self.action_output, message.content)
        self.assertEqual(TrajectoryState.COMPLETED, self.env.trajectory[0].status)

    @pytest.mark.asyncio
    async def test_output_extraction(self):
        """Test extracting the output from the workflow execution"""
        # Set up the llm.async_generate to return an output
        output_parser = MockLLMOutputParser()
        self.mock_llm.async_generate.return_value = output_parser
        
        # Complete all tasks in the workflow
        self.workflow.set_node_status("Task1", WorkFlowNodeState.COMPLETED)
        self.workflow.set_node_status("Task2", WorkFlowNodeState.COMPLETED)
        self.workflow.set_node_status("Task3", WorkFlowNodeState.COMPLETED)
        
        # Add task messages to the environment
        for task_name in ["Task1", "Task2", "Task3"]:
            message = Message(
                content="Task output",
                agent="TestAgent",
                action="TestAction",
                prompt="Test prompt",
                msg_type=MessageType.RESPONSE,
                wf_goal="Test Workflow",
                wf_task=task_name
            )
            self.env.update(message=message, state=TrajectoryState.COMPLETED)
        
        # Run the test
        output = await self.workflow_manager.extract_output(
            graph=self.workflow, 
            env=self.env
        )
        
        # Check results
        self.assertEqual("Test output", output)
        
        # Verify the LLM was called
        self.mock_llm.async_generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_edge_case_handling(self):
        """Test edge case handling in workflow management"""
        # Test case: No tasks available for scheduling
        # Complete all tasks to ensure no new tasks are available
        self.workflow.set_node_status("Task1", WorkFlowNodeState.COMPLETED)
        self.workflow.set_node_status("Task2", WorkFlowNodeState.COMPLETED)
        self.workflow.set_node_status("Task3", WorkFlowNodeState.COMPLETED)
        
        # Try to schedule the next task
        task = await self.workflow_manager.schedule_next_task(graph=self.workflow, env=self.env)
        
        # Check that no task was scheduled
        self.assertIsNone(task)
        
        # Reset the workflow
        self.workflow.reset_graph()
        
        # Test case: Only one candidate task is available
        # This is tested in test_sync_task_scheduling_with_single_task
        
        # Test case: Task with no agents
        # Create a task with no agents
        task_no_agents = WorkFlowNode(
            name="TaskNoAgents",
            description="Task with no agents",
            inputs=[],
            outputs=[],
            agents=[],
            status=WorkFlowNodeState.PENDING
        )
        
        # Create a mock agent manager
        mock_agent_manager = Mock()
        
        # Try to schedule an action for this task
        with self.assertRaises(ValueError):
            await self.workflow_manager.schedule_next_action(
                goal="Test Goal", 
                task=task_no_agents, 
                agent_manager=mock_agent_manager, 
                env=self.env
            )


if __name__ == "__main__":
    unittest.main() 