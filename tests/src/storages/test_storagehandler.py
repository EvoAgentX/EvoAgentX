import unittest
import json
from datetime import datetime
from unittest.mock import patch

from evoagentx.storages.storages_config import StoreConfig, DBConfig, VectorStoreConfig
from evoagentx.storages.base import StorageHandler
from evoagentx.storages.schema import TableType


class TestStorageHandler(unittest.TestCase):
    """
    Test suite for StorageHandler's database operations on Workflow, Agent, and History.
    Uses an in-memory SQLite database for isolated testing.
    """
    def setUp(self):
        """
        Set up the test environment by initializing StorageHandler with an in-memory SQLite database.
        """
        # Mock configuration
        db_config = DBConfig(db_name="sqlite", path=":memory:")
        store_config = StoreConfig(dbConfig=db_config)
        
        self.storage = StorageHandler(storageConfig=store_config)
        
        # Sample data for testing
        self.agent_data = {
            "name": "test_agent",
            "content": {"role": "assistant", "settings": {"active": True}},
            "date": "2025-05-13"
        }
        self.workflow_data = {"name": "test_workflow", "content": {
    "class_name": "WorkFlowGraph",
    "goal": "Generate html code for the Tetris game that can be played in the browser.",
    "nodes": [
        {
            "class_name": "WorkFlowNode",
            "name": "game_structure_design",
            "description": "Create an outline of the Tetris game's structure, including the main game area, score display, and control buttons.",
            "inputs": [
                {
                    "class_name": "Parameter",
                    "name": "goal",
                    "type": "string",
                    "description": "The user's goal in textual format.",
                    "required": True
                }
            ],
            "outputs": [
                {
                    "class_name": "Parameter",
                    "name": "html_structure",
                    "type": "string",
                    "description": "The basic HTML structure outlining the game area, score display, and buttons.",
                    "required": True
                }
            ],
            "reason": "This sub-task establishes the foundational layout required for a functional Tetris game in HTML.",
            "agents": [
                {
                    "name": "tetris_game_structure_agent",
                    "description": "This agent creates the basic HTML structure for the Tetris game, including the game area, score display, and control buttons based on the user's goal.",
                    "inputs": [
                        {
                            "name": "goal",
                            "type": "string",
                            "description": "The user's goal in textual format.",
                            "required": True
                        }
                    ],
                    "outputs": [
                        {
                            "name": "html_structure",
                            "type": "string",
                            "description": "The basic HTML structure outlining the game area, score display, and buttons.",
                            "required": True
                        }
                    ],
                    "prompt": "### Objective\nCreate the basic HTML structure for a Tetris game, incorporating the main game area, score display, and control buttons based on the user's goal.\n\n### Instructions\n1. Read the user's goal: <input>{goal}</input>\n2. Design the main game area where the Tetris pieces will fall.\n3. Create an element to display the current score.\n4. Include buttons to control the game (e.g., start, pause, reset).\n5. Assemble these elements into a coherent HTML structure that can be utilized in a web environment.\n6. Output the generated HTML structure.\n\n### Output Format\nYour final output should ALWAYS in the following format:\n\n## Thought\nBriefly explain the reasoning process for creating the HTML structure of the Tetris game.\n\n## html_structure\nThe basic HTML structure outlining the game area, score display, and buttons."
                }
            ],
            "status": "pending"
        },
        {
            "class_name": "WorkFlowNode",
            "name": "style_application",
            "description": "Add CSS styles to the HTML structure for visual aesthetics and layout to make the game look visually appealing.",
            "inputs": [
                {
                    "class_name": "Parameter",
                    "name": "html_structure",
                    "type": "string",
                    "description": "The basic HTML structure of the Tetris game.",
                    "required": True
                }
            ],
            "outputs": [
                {
                    "class_name": "Parameter",
                    "name": "styled_game",
                    "type": "string",
                    "description": "The styled HTML code that includes CSS for the Tetris game.",
                    "required": True
                }
            ],
            "reason": "Styling is essential for enhancing the user experience and ensuring the game is visually organized and engaging.",
            "agents": [
                {
                    "name": "css_style_application_agent",
                    "description": "This agent applies CSS styles to the given HTML structure to create a visually appealing layout for the Tetris game.",
                    "inputs": [
                        {
                            "name": "html_structure",
                            "type": "string",
                            "description": "The basic HTML structure of the Tetris game.",
                            "required": True
                        }
                    ],
                    "outputs": [
                        {
                            "name": "styled_game",
                            "type": "string",
                            "description": "The styled HTML code that includes CSS for the Tetris game.",
                            "required": True
                        }
                    ],
                    "prompt": "### Objective\nEnhance the provided HTML structure by applying CSS styles to create a visually appealing layout for the Tetris game.\n\n### Instructions\n1. Begin with the provided HTML structure: <input>{html_structure}</input>\n2. Analyze the elements in the HTML to decide the appropriate CSS styles that will enhance its appearance.\n3. Write CSS styles that cater to visual aesthetics such as colors, fonts, borders, and spacing.\n4. Integrate the CSS styles into the HTML structure properly.\n5. Ensure the output is a well-formatted HTML document that includes the applied CSS styles.\n\n### Output Format\nYour final output should ALWAYS in the following format:\n\n## Thought\nBriefly explain the reasoning process for achieving the objective.\n\n## styled_game\nThe styled HTML code that includes CSS for the Tetris game."
                }
            ],
            "status": "pending"
        },
        {
            "class_name": "WorkFlowNode",
            "name": "game_logic_implementation",
            "description": "Implement the JavaScript logic for the Tetris game, including piece movement, collision detection, and score tracking.",
            "inputs": [
                {
                    "class_name": "Parameter",
                    "name": "styled_game",
                    "type": "string",
                    "description": "The styled HTML code for the Tetris game.",
                    "required": True
                }
            ],
            "outputs": [
                {
                    "class_name": "Parameter",
                    "name": "complete_game_code",
                    "type": "string",
                    "description": "The complete HTML, CSS, and JavaScript code for a functional Tetris game.",
                    "required": True
                }
            ],
            "reason": "This sub-task is crucial for making the game interactive and functional, allowing users to play.",
            "agents": [
                {
                    "name": "tetris_logic_agent",
                    "description": "This agent implements the JavaScript logic required for the Tetris game, ensuring piece movements, collision detection, and score tracking functionalities are properly integrated.",
                    "inputs": [
                        {
                            "name": "styled_game",
                            "type": "string",
                            "description": "The styled HTML code for the Tetris game.",
                            "required": True
                        }
                    ],
                    "outputs": [
                        {
                            "name": "complete_game_code",
                            "type": "string",
                            "description": "The complete HTML, CSS, and JavaScript code for a functional Tetris game.",
                            "required": True
                        }
                    ],
                    "prompt": "### Objective\nImplement the JavaScript logic for the Tetris game, ensuring functionalities for piece movement, collision detection, and score tracking are included in the output.\n\n### Instructions\n1. Analyze the styled HTML code provided: <input>{styled_game}</input>\n2. Develop JavaScript functions that handle the movement of Tetris pieces, including left, right, and rotation controls.\n3. Implement collision detection logic to ensure pieces do not fall through the bottom or collide with existing pieces.\n4. Create a scoring system that tracks the player's progress and updates the score based on cleared lines.\n5. Combine the JavaScript logic with the existing styled HTML to create a complete game code output.\n\n### Output Format\nYour final output should ALWAYS in the following format:\n\n## Thought\nBriefly explain the reasoning process for implementing the game logic for Tetris.\n\n## complete_game_code\nThe completed HTML, CSS, and JavaScript code for a functional Tetris game."
                }
            ],
            "status": "pending"
        },
        {
            "class_name": "WorkFlowNode",
            "name": "testing_and_refinement",
            "description": "Test the generated Tetris game for bugs and usability issues, refining the code as necessary.",
            "inputs": [
                {
                    "class_name": "Parameter",
                    "name": "complete_game_code",
                    "type": "string",
                    "description": "The complete HTML, CSS, and JavaScript code for the Tetris game.",
                    "required": True
                }
            ],
            "outputs": [
                {
                    "class_name": "Parameter",
                    "name": "final_output",
                    "type": "string",
                    "description": "The final tested and refined code for the Tetris game.",
                    "required": True
                }
            ],
            "reason": "Testing is vital to ensure that the game functions correctly across different browsers and provides a smooth user experience.",
            "agents": [
                {
                    "name": "tetris_game_testing_agent",
                    "description": "This agent tests the generated Tetris game code for functionality, identifies bugs, and provides refinements as needed to ensure smooth gameplay and usability.",
                    "inputs": [
                        {
                            "name": "complete_game_code",
                            "type": "string",
                            "description": "The complete HTML, CSS, and JavaScript code for the Tetris game.",
                            "required": True
                        }
                    ],
                    "outputs": [
                        {
                            "name": "final_output",
                            "type": "string",
                            "description": "The final tested and refined code for the Tetris game.",
                            "required": True
                        }
                    ],
                    "prompt": "### Objective\nTest the complete Tetris game code for bugs and usability issues, and refine the code as necessary for improved performance.\n\n### Instructions\n1. Load the complete game code: <input>{complete_game_code}</input> into a browser.\n2. Test the game functionality, focusing on user controls, collision detection, and game progression.\n3. Identify any bugs or usability issues that arise during testing.\n4. Document the identified issues and make necessary adjustments to the code to resolve them.\n5. Ensure that the final code adheres to best practices for HTML, CSS, and JavaScript.\n6. Output the refined and tested code as the final result.\n\n### Output Format\nYour final output should ALWAYS in the following format:\n\n## Thought\nBriefly explain the reasoning process for testing and refining the Tetris game code.\n\n## final_output\nThe final tested and refined code for the Tetris game."
                }
            ],
            "status": "pending"
        }
    ],
    "edges": [
        {
            "class_name": "WorkFlowEdge",
            "source": "game_structure_design",
            "target": "style_application",
            "priority": 0
        },
        {
            "class_name": "WorkFlowEdge",
            "source": "style_application",
            "target": "game_logic_implementation",
            "priority": 0
        },
        {
            "class_name": "WorkFlowEdge",
            "source": "game_logic_implementation",
            "target": "testing_and_refinement",
            "priority": 0
        }
    ],
    "graph": None
},
"date": "2025-05-13"}

        self.history_data = {
            "memory_id": "mem_001",
            "old_memory": "Initial content",
            "new_memory": "Updated content",
            "event": "update",
            "created_at": "2025-05-13T09:00:00",
            "updated_at": "2025-05-13T09:30:00"
        }

    def test_save_and_load_agent(self):
        """
        Test saving and loading an agent, verifying data integrity and JSON parsing.
        """
        # Save agent
        self.storage.save_agent(self.agent_data)
        self.storage.save_agent(self.agent_data, "nihao")
        
        # Load agent
        loaded = self.storage.load_agent("test_agent")
        
        # Verify data
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["name"], "test_agent")
        self.assertEqual(loaded["content"], self.agent_data["content"])  # JSON parsed
        self.assertEqual(loaded["date"], "2025-05-13")

    def test_save_and_load_workflow(self):
        """
        Test saving and loading a workflow, verifying data integrity and JSON parsing.
        """
        # Save workflow
        self.storage.save_workflow(self.workflow_data)
        
        # Load workflow
        loaded = self.storage.load_workflow("test_workflow")
        
        # Verify data
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["name"], "test_workflow")
        self.assertEqual(loaded["content"], self.workflow_data["content"])  # JSON parsed
        self.assertEqual(loaded["date"], "2025-05-13")

    def test_save_and_load_history(self):
        """
        Test saving and loading a history entry, verifying data integrity.
        """
        # Save history
        self.storage.save_history(self.history_data)
        
        # Load history
        loaded = self.storage.load_history("mem_001")
        
        # Verify data
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["memory_id"], "mem_001")
        self.assertEqual(loaded["old_memory"], "Initial content")
        self.assertEqual(loaded["new_memory"], "Updated content")
        self.assertEqual(loaded["event"], "update")
        self.assertEqual(loaded["created_at"], "2025-05-13T09:00:00")
        self.assertEqual(loaded["updated_at"], "2025-05-13T09:30:00")

    def test_load_non_existent_agent(self):
        """
        Test loading a non-existent agent returns None.
        """
        loaded = self.storage.load_agent("non_existent_agent")
        self.assertIsNone(loaded)

    def test_load_non_existent_workflow(self):
        """
        Test loading a non-existent workflow returns None.
        """
        loaded = self.storage.load_workflow("non_existent_workflow")
        self.assertIsNone(loaded)

    def test_load_non_existent_history(self):
        """
        Test loading a non-existent history entry returns None.
        """
        loaded = self.storage.load_history("non_existent_mem")
        self.assertIsNone(loaded)

    def test_save_invalid_agent(self):
        """
        Test saving an agent without a 'name' field raises ValueError.
        """
        invalid_data = {"content": {"role": "assistant"}, "date": "2025-05-13"}
        with self.assertRaises(ValueError):
            self.storage.save_agent(invalid_data)

    def test_save_invalid_workflow(self):
        """
        Test saving a workflow without a 'name' field raises ValueError.
        """
        invalid_data = {"content": {"steps": ["step1"]}, "date": "2025-05-13"}
        with self.assertRaises(ValueError):
            self.storage.save_workflow(invalid_data)

    def test_save_invalid_history(self):
        """
        Test saving a history entry without a 'memory_id' field raises ValueError.
        """
        invalid_data = {
            "old_memory": "Initial",
            "new_memory": "Updated",
            "event": "update"
        }
        with self.assertRaises(ValueError):
            self.storage.save_history(invalid_data)

    def test_remove_agent(self):
        """
        Test removing an agent and verify it's no longer loadable.
        """
        # Save and remove agent
        self.storage.save_agent(self.agent_data)
        self.storage.remove_agent("test_agent")
        
        # Verify agent is gone
        loaded = self.storage.load_agent("test_agent")
        self.assertIsNone(loaded)

    def test_remove_non_existent_agent(self):
        """
        Test removing a non-existent agent raises ValueError.
        """
        with self.assertRaises(ValueError):
            self.storage.remove_agent("non_existent_agent")

    def test_update_agent(self):
        """
        Test updating an existing agent's data.
        """
        # Save initial agent
        self.storage.save_agent(self.agent_data)
        
        # Update agent data
        updated_data = {
            "name": "test_agent",
            "content": {"role": "admin", "settings": {"active": False}},
            "date": "2025-05-14"
        }
        self.storage.save_agent(updated_data)
        
        # Load and verify updated data
        loaded = self.storage.load_agent("test_agent")
        self.assertEqual(loaded["content"], updated_data["content"])
        self.assertEqual(loaded["date"], "2025-05-14")

    def test_update_workflow(self):
        """
        Test updating an existing workflow's data.
        """
        # Save initial workflow
        self.storage.save_workflow(self.workflow_data)
        
        # Update workflow data
        updated_data ={"name": "test_workflow", "content": {"test": True}, "date": "2025-05-15"}
        self.storage.save_workflow(updated_data)
        
        # Load and verify updated data
        loaded = self.storage.load_workflow("test_workflow")
        self.assertEqual(loaded["content"], updated_data["content"])
        self.assertEqual(loaded["date"], "2025-05-15")

    def test_update_history(self):
        """
        Test updating an existing history entry.
        """
        # Save initial history
        self.storage.save_history(self.history_data)
        
        # Update history data
        updated_data = {
            "memory_id": "mem_001",
            "old_memory": "Initial content",
            "new_memory": "Further updated content",
            "event": "modify",
            "created_at": "2025-05-13T09:00:00",
            "updated_at": "2025-05-13T10:00:00"
        }
        self.storage.save_history(updated_data)
        
        # Load and verify updated data
        loaded = self.storage.load_history("mem_001")
        self.assertEqual(loaded["new_memory"], "Further updated content")
        self.assertEqual(loaded["event"], "modify")
        self.assertEqual(loaded["updated_at"], "2025-05-13T10:00:00")

    def test_bulk_save_and_load(self):
        """
        Test saving multiple records to all tables and loading them.
        """
        # Prepare bulk data
        agent_data2 = {
            "name": "test_agent2",
            "content": {"role": "user", "settings": {"active": True}},
            "date": "2025-05-13"
        }
        workflow_data2 = {
            "name": "test_workflow2",
            "content": {"steps": ["stepA", "stepB"], "config": {"timeout": 45}},
            "date": "2025-05-13"
        }
        history_data2 = {
            "memory_id": "mem_002",
            "old_memory": "Old content",
            "new_memory": "New content",
            "event": "create",
            "created_at": "2025-05-13T10:00:00",
            "updated_at": "2025-05-13T10:00:00"
        }
        
        # Save bulk data
        bulk_data = {
            TableType.store_agent.value: [self.agent_data, agent_data2],
            TableType.store_workflow.value: [self.workflow_data, workflow_data2],
            TableType.store_history.value: [self.history_data, history_data2]
        }
        self.storage.save(bulk_data)
        
        # Load all data
        all_data = self.storage.load()
        
        # Verify data presence
        self.assertIn(TableType.store_agent.value, all_data)
        self.assertIn(TableType.store_workflow.value, all_data)
        self.assertIn(TableType.store_history.value, all_data)
        self.assertEqual(len(all_data[TableType.store_agent.value]), 2)
        self.assertEqual(len(all_data[TableType.store_workflow.value]), 2)
        self.assertEqual(len(all_data[TableType.store_history.value]), 2)
        
        # Verify specific records
        agent_names = [record["name"] for record in all_data[TableType.store_agent.value]]
        self.assertIn("test_agent", agent_names)
        self.assertIn("test_agent2", agent_names)
        
        workflow_names = [record["name"] for record in all_data[TableType.store_workflow.value]]
        self.assertIn("test_workflow", workflow_names)
        self.assertIn("test_workflow2", workflow_names)
        
        history_ids = [record["memory_id"] for record in all_data[TableType.store_history.value]]
        self.assertIn("mem_001", history_ids)
        self.assertIn("mem_002", history_ids)

    def test_save_invalid_table(self):
        """
        Test saving data to an unknown table raises ValueError.
        """
        invalid_data = {"unknown_table": [self.agent_data]}
        with self.assertRaises(ValueError):
            self.storage.save(invalid_data)

    def tearDown(self):
        """
        Clean up by closing the database connection.
        """
        self.storage.storageDB.connection.close()

if __name__ == "__main__":
    unittest.main()