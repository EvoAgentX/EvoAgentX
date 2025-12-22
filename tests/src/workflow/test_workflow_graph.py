import unittest
from evoagentx.core.base_config import Parameter
from evoagentx.workflow.workflow_graph import WorkFlowNode, WorkFlowGraph, WorkFlowEdge, WorkFlowNodeState


class TestWorkFlowGraph(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        # Create nodes for testing
        self.task1 = WorkFlowNode(
            name="Task1",
            description="First task",
            inputs=[Parameter(name="input1", type="string", description="Input 1")],
            outputs=[Parameter(name="output1", type="string", description="Output 1")],
            agents=["TestAgent"],
            status=WorkFlowNodeState.PENDING
        )
        
        self.task2 = WorkFlowNode(
            name="Task2",
            description="Second task",
            inputs=[Parameter(name="output1", type="string", description="Output from Task1")],
            outputs=[Parameter(name="output2", type="string", description="Output 2")],
            agents=["TestAgent"],
            status=WorkFlowNodeState.PENDING
        )
        
        self.task3 = WorkFlowNode(
            name="Task3",
            description="Third task",
            inputs=[Parameter(name="output2", type="string", description="Output from Task2")],
            outputs=[Parameter(name="final_output", type="string", description="Final output")],
            agents=["TestAgent"],
            status=WorkFlowNodeState.PENDING
        )
        
        # Create a fork-join workflow structure
        #     Task1
        #    /     
        # Task2 -- Task3
        #    \     /
        #    Task4
        self.task4 = WorkFlowNode(
            name="Task4",
            description="Fourth task (join)",
            inputs=[
                Parameter(name="output2", type="string", description="Output from Task2"),
                Parameter(name="final_output", type="string", description="Output from Task3")
            ],
            outputs=[Parameter(name="result", type="string", description="Final result")],
            agents=["TestAgent"],
            status=WorkFlowNodeState.PENDING
        )
        
        # Create a simple linear workflow
        self.linear_graph = WorkFlowGraph(
            goal="Simple Linear Workflow",
            nodes=[self.task1, self.task2, self.task3],
            edges=[
                WorkFlowEdge(source="Task1", target="Task2"),
                WorkFlowEdge(source="Task2", target="Task3")
            ]
        )
        
        # Create a fork-join workflow
        self.fork_join_graph = WorkFlowGraph(
            goal="Fork-Join Workflow",
            nodes=[self.task1, self.task2, self.task3, self.task4],
            edges=[
                WorkFlowEdge(source="Task1", target="Task2"),
                WorkFlowEdge(source="Task2", target="Task3"),
                WorkFlowEdge(source="Task2", target="Task4"),
                WorkFlowEdge(source="Task3", target="Task4")
            ]
        )
        
        # Create a workflow with a cycle
        self.cycle_graph = WorkFlowGraph(
            goal="Workflow with Cycle",
            nodes=[self.task1, self.task2, self.task3],
            edges=[
                WorkFlowEdge(source="Task1", target="Task2"),
                WorkFlowEdge(source="Task2", target="Task3"),
                WorkFlowEdge(source="Task3", target="Task2")  # Creates a cycle
            ]
        )
    
    def test_graph_initialization(self):
        """Test that graph is correctly initialized with nodes and edges."""
        # Check linear graph
        self.assertEqual(3, len(self.linear_graph.nodes))
        self.assertEqual(2, len(self.linear_graph.edges))
        
        # Verify node order
        self.assertEqual("Task1", self.linear_graph.nodes[0].name)
        self.assertEqual("Task2", self.linear_graph.nodes[1].name)
        self.assertEqual("Task3", self.linear_graph.nodes[2].name)
        
        # Verify edge connections
        edge_pairs = [(edge.source, edge.target) for edge in self.linear_graph.edges]
        self.assertIn(("Task1", "Task2"), edge_pairs)
        self.assertIn(("Task2", "Task3"), edge_pairs)
    
    def test_find_initial_nodes(self):
        """Test finding initial nodes in a workflow."""
        # In the linear graph, Task1 is the only initial node
        initial_nodes = self.linear_graph.find_initial_nodes()
        self.assertEqual(1, len(initial_nodes))
        self.assertEqual("Task1", initial_nodes[0])
        
        # In the fork-join graph, Task1 is the only initial node
        initial_nodes = self.fork_join_graph.find_initial_nodes()
        self.assertEqual(1, len(initial_nodes))
        self.assertEqual("Task1", initial_nodes[0])
        
        # In the cycle graph, Task1 is not an initial node because of the cycle
        # All nodes have incoming edges in a cycle
        initial_nodes = self.cycle_graph.find_initial_nodes()
        self.assertEqual(1, len(initial_nodes))
    
    def test_find_end_nodes(self):
        """Test finding end nodes in a workflow."""
        # In the linear graph, Task3 is the only end node
        end_nodes = self.linear_graph.find_end_nodes()
        self.assertEqual(1, len(end_nodes))
        self.assertEqual("Task3", end_nodes[0])
        
        # In the fork-join graph, Task4 is the only end node
        end_nodes = self.fork_join_graph.find_end_nodes()
        self.assertEqual(1, len(end_nodes))
        self.assertEqual("Task4", end_nodes[0])
        
        # In the cycle graph, there are no end nodes because of the cycle
        # All nodes have outgoing edges in a cycle
        end_nodes = self.cycle_graph.find_end_nodes()
        self.assertEqual(0, len(end_nodes))
    
    def test_next_execution(self):
        """Test the 'next' method to determine the next executable tasks."""
        # In the linear graph, initially Task1 is the only executable task
        next_tasks = self.linear_graph.next()
        self.assertEqual(1, len(next_tasks))
        self.assertEqual("Task1", next_tasks[0].name)
        
        # After completing Task1, Task2 should be the next executable task
        self.linear_graph.set_node_status("Task1", WorkFlowNodeState.COMPLETED)
        next_tasks = self.linear_graph.next()
        self.assertEqual(1, len(next_tasks))
        self.assertEqual("Task2", next_tasks[0].name)
        
        # After completing Task2, Task3 should be the next executable task
        self.linear_graph.set_node_status("Task2", WorkFlowNodeState.COMPLETED)
        next_tasks = self.linear_graph.next()
        self.assertEqual(1, len(next_tasks))
        self.assertEqual("Task3", next_tasks[0].name)
        
        # After completing Task3, there should be no more executable tasks
        self.linear_graph.set_node_status("Task3", WorkFlowNodeState.COMPLETED)
        next_tasks = self.linear_graph.next()
        self.assertEqual(0, len(next_tasks))
    
    def test_fork_join_execution(self):
        """Test execution in a fork-join workflow."""
        # Initially Task1 is the only executable task
        next_tasks = self.fork_join_graph.next()
        self.assertEqual(1, len(next_tasks))
        self.assertEqual("Task1", next_tasks[0].name)
        
        # After completing Task1, both Task2 and Task3 should be executable
        self.fork_join_graph.set_node_status("Task1", WorkFlowNodeState.COMPLETED)
        next_tasks = self.fork_join_graph.next()
        self.assertEqual(1, len(next_tasks))
        self.assertEqual("Task2", next_tasks[0].name)
        
        # After completing Task2 but not Task3, Task4 should not be executable yet
        self.fork_join_graph.set_node_status("Task2", WorkFlowNodeState.COMPLETED)
        next_tasks = self.fork_join_graph.next()
        self.assertEqual(1, len(next_tasks))
        self.assertEqual("Task3", next_tasks[0].name)
        
        # After completing Task3 as well, Task4 should be executable
        self.fork_join_graph.set_node_status("Task3", WorkFlowNodeState.COMPLETED)
        next_tasks = self.fork_join_graph.next()
        self.assertEqual(1, len(next_tasks))
        self.assertEqual("Task4", next_tasks[0].name)
        
        # After completing Task4, there should be no more executable tasks
        self.fork_join_graph.set_node_status("Task4", WorkFlowNodeState.COMPLETED)
        next_tasks = self.fork_join_graph.next()
        self.assertEqual(0, len(next_tasks))
    
    def test_cycle_detection(self):
        """Test cycle detection in a workflow."""
        # The cycle graph should identify a loop
        loops = self.cycle_graph._find_all_loops()
        self.assertTrue(loops)  # Should contain at least one loop
        
        # Check if Task1 is identified as both a loop start and loop end
        self.assertTrue(self.cycle_graph.is_loop_start("Task2"))
        self.assertTrue(self.cycle_graph.is_loop_end("Task3"))
    
    def test_node_status_management(self):
        """Test node status management."""
        # Check initial status
        self.assertEqual(WorkFlowNodeState.PENDING, self.linear_graph.get_node_status("Task1"))
        
        # Set status to RUNNING
        self.linear_graph.set_node_status("Task1", WorkFlowNodeState.RUNNING)
        self.assertEqual(WorkFlowNodeState.RUNNING, self.linear_graph.get_node_status("Task1"))
        self.assertTrue(self.linear_graph.running("Task1"))
        
        # Set status to COMPLETED
        self.linear_graph.set_node_status("Task1", WorkFlowNodeState.COMPLETED)
        self.assertEqual(WorkFlowNodeState.COMPLETED, self.linear_graph.get_node_status("Task1"))
        self.assertTrue(self.linear_graph.completed("Task1"))
        
        # Set status to FAILED
        self.linear_graph.set_node_status("Task1", WorkFlowNodeState.FAILED)
        self.assertEqual(WorkFlowNodeState.FAILED, self.linear_graph.get_node_status("Task1"))
        self.assertTrue(self.linear_graph.failed("Task1"))
    
    def test_graph_reset(self):
        """Test resetting the graph to initial state."""
        # Set all nodes to COMPLETED
        for node in self.linear_graph.nodes:
            self.linear_graph.set_node_status(node.name, WorkFlowNodeState.COMPLETED)
        
        # Verify all nodes are COMPLETED
        for node in self.linear_graph.nodes:
            self.assertEqual(WorkFlowNodeState.COMPLETED, node.status)
        
        # Reset the graph
        self.linear_graph.reset_graph()
        
        # Verify all nodes are reset to PENDING
        for node in self.linear_graph.nodes:
            self.assertEqual(WorkFlowNodeState.PENDING, node.status)
    
    def test_graph_dependency_checking(self):
        """Test checking dependencies between nodes."""
        # In the linear graph, Task2 depends on Task1
        self.assertFalse(self.linear_graph.are_dependencies_complete("Task2"))
        
        # After completing Task1, Task2's dependencies should be satisfied
        self.linear_graph.set_node_status("Task1", WorkFlowNodeState.COMPLETED)
        self.assertTrue(self.linear_graph.are_dependencies_complete("Task2"))
        
        # In the fork-join graph, Task4 depends on both Task2 and Task3
        self.assertFalse(self.fork_join_graph.are_dependencies_complete("Task4"))
        
        # After completing only Task2, Task4's dependencies are still not satisfied
        self.fork_join_graph.set_node_status("Task1", WorkFlowNodeState.COMPLETED)
        self.fork_join_graph.set_node_status("Task2", WorkFlowNodeState.COMPLETED)
        self.assertFalse(self.fork_join_graph.are_dependencies_complete("Task4"))
        
        # After completing Task3 as well, Task4's dependencies should be satisfied
        self.fork_join_graph.set_node_status("Task3", WorkFlowNodeState.COMPLETED)
        self.assertTrue(self.fork_join_graph.are_dependencies_complete("Task4"))


if __name__ == "__main__":
    unittest.main()