import asyncio
import json
import os
import time
from collections.abc import Callable
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import Field, PositiveInt

from ..actions.agent_generation import (
    AgentGenerationOutput,
    GeneratedAgent,
    SelectedAgent,
)
from ..actions.task_planning import TaskPlanningOutput
from ..agents.agent import Agent
from ..agents.agent_adaptor import AgentAdaptor
from ..agents.agent_generator import AgentGenerator
from ..agents.task_planner import TaskPlanner
from ..agents.workflow_reviewer import WorkFlowReviewer
from ..core.base_config import Parameter
from ..core.logging import logger
from ..core.message import Message, MessageType
from ..core.module import BaseModule
from ..models.base_model import BaseLLM
from ..rag.rag import RAGEngine
from ..rag.rag_config import EmbeddingConfig, RAGConfig, RetrievalConfig
from ..rag.schema import Chunk, ChunkMetadata, Corpus
from ..storages.base import StorageHandler
from ..storages.storages_config import DBConfig, StoreConfig, VectorStoreConfig
from ..tools.tool import Tool, Toolkit
from ..utils.utils import recursive_remove
from ..workflow.workflow_graph import WorkFlowEdge, WorkFlowGraph, WorkFlowNode


class WorkFlowGenerator(BaseModule):
    """
    Automated workflow generation system based on high-level goals.
    
    The WorkFlowGenerator is responsible for creating complete workflow graphs
    from high-level goals or task descriptions. It breaks down the goal into
    subtasks, creates the necessary dependency connections between tasks,
    and assigns or generates appropriate agents for each task.
    
    Attributes:
        llm: Language model used for generation and planning
        task_planner: Component responsible for breaking down goals into subtasks
        agent_generator: Component responsible for agent assignment or creation
        workflow_reviewer: Component for reviewing and improving workflows
        num_turns: Number of refinement iterations for the workflow
        tools: A list of tools that can be used in the workflow.
        prebuilt_agents: A list of prebuilt agents that can be used in the workflow.
        rag_engine: Retrieve similar workflows from the database to use as examples for generating the new workflow.
        workflow_folder: Folder that contains the workflow json files.
        db_path: Path to the SQLite database file that contains the embeddings.
    """
    llm: Optional[BaseLLM] = None
    task_planner: Optional[TaskPlanner] = Field(default=None, description="Responsible for breaking down the high-level task into manageable sub-tasks.")
    agent_generator: Optional[AgentGenerator] = Field(default=None, description="Assigns or generates the appropriate agent(s) to handle each sub-task.")
    workflow_reviewer: Optional[WorkFlowReviewer] = Field(default=None, description="Provides feedback and reflections to improve the generated workflow.")
    num_turns: Optional[PositiveInt] = Field(default=0, description="Specifies the number of refinement iterations for the generated workflow.")
    tools: Optional[List[Union[Toolkit, Tool]]] = Field(default=None, description="A list of tools that can be used in the workflow.")
    prebuilt_agents: Optional[List[Agent]] = Field(default=None, description="A list of prebuilt agents that can be used in the workflow.")
    rag_engine: Optional[RAGEngine] = Field(default=None, description="An optional RAG engine to retrieve similar workflows from database to use as examples in workflow generation. If provided, overwrites the default RAG engine.")
    workflow_folder: str = Field(default="./workflows", description="Folder that contains the workflow json files. These workflows will be used as examples in workflow generation (RAG).")
    db_path: Optional[str] = Field(default=None, description="Path to an optional SQLite database file that contains the embeddings of workflow examples. If provided, examples in the database will be used in workflow generation (RAG).")

    def init_module(self):
        if self.task_planner is None:
            if self.llm is None:
                raise ValueError("Must provide `llm` when `task_planner` is None")
            self.task_planner = TaskPlanner(llm=self.llm)
        
        if self.agent_generator is None:
            if self.llm is None:
                raise ValueError("Must provide `llm` when `agent_generator` is None")
            self.agent_generator = AgentGenerator(llm=self.llm, tools=self.tools)
        
        # TODO add WorkFlowReviewer
        # if self.workflow_reviewer is None:
        #     if self.llm is None:
        #         raise ValueError(f"Must provide `llm` when `workflow_reviewer` is None")
        #     self.workflow_reviewer = WorkFlowReviewer(llm=self.llm)

        self._prepare_rag_engine()

        self.prebuilt_agents_map = None
        if self.prebuilt_agents is not None:
            self.prebuilt_agents_map = {agent.name: agent for agent in self.prebuilt_agents}
        self.agents_rag_engine = None

        self.agent_adaptor_names = []


    def _execute_with_retry(
        self, 
        operation_name: str, 
        operation: Callable, 
        retries_left: int = 1, 
        **kwargs
    ) -> tuple[Any, int]:
        """Helper method to execute operations with retry logic.
        
        Args:
            operation_name: Name of the operation for logging
            operation: Callable that performs the operation
            retries_left: Number of retry attempts remaining
            **kwargs: Additional arguments to pass to the operation
            
        Returns:
            Tuple of (operation_result, number_of_retries_used)
            
        Raises:
            ValueError: If operation fails after all retries are exhausted
        """
        cur_retries = 0

        while cur_retries <= retries_left:  # Changed < to <= to include the initial try
            try:
                logger.info(f"{operation_name} (attempt {cur_retries + 1}/{retries_left + 1}) ...")
                result = operation(**kwargs)
                return result, cur_retries
            except Exception as e:
                if cur_retries == retries_left:
                    raise ValueError(f"Failed to {operation_name} after {cur_retries + 1} attempts.\nError: {e}") from e
                sleep_time = 2 ** cur_retries
                logger.exception(f"Failed to {operation_name} in {cur_retries + 1} attempts. Retry after {sleep_time} seconds.\nError: {e}")
                time.sleep(sleep_time)
                cur_retries += 1


    def generate_workflow(
        self, 
        goal: str, 
        workflow_inputs: List[Parameter] = [Parameter(name="workflow_input", type="string", description="workflow input")], 
        workflow_outputs: List[Parameter] = [Parameter(name="workflow_output", type="string", description="workflow output")], 
        retry: int = 1, 
        **kwargs
    ) -> WorkFlowGraph:
        """
        Generate a workflow that can be used to achieve the given goal.

        Args:
            goal (str): The goal to generate a workflow for
            workflow_inputs (List[Parameter]): List of workflow inputs
            workflow_outputs (List[Parameter]): List of workflow outputs
            retry (int): Number of retry attempts
            **kwargs: Additional arguments to pass to the operation

        Returns:
            WorkFlowGraph: The generated workflow graph
        """

        # Validate input
        if not goal or len(goal.strip()) < 10:
            raise ValueError("Goal must be at least 10 characters and descriptive")

        plan_history, plan_suggestion = "", ""
        task_examples, agent_examples = None, None

        # retrieve workflow examples if rag_engine is provided
        if self.rag_engine is not None:
            task_examples, agent_examples = self._get_examples(goal)

        # Generate the initial workflow plan
        cur_retries = 0
        workflow, added_retries = self._execute_with_retry(
            operation_name="Generating a workflow plan",
            operation=self.generate_workflow_plan,
            retries_left=retry,
            goal=goal,
            workflow_inputs=workflow_inputs,
            workflow_outputs=workflow_outputs,
            history=plan_history,
            suggestion=plan_suggestion,
            examples=task_examples,
        )
        cur_retries += added_retries

        # generate / assigns the initial agents
        logger.info("Generating agents for the workflow ...")
        workflow, added_retries = self._execute_with_retry(
            operation_name="Generating agents for the workflow",
            operation=self.generate_agents,
            retries_left=retry - cur_retries,
            workflow=workflow,
            examples=agent_examples
        )

        self.agents_rag_engine = None
        return workflow


    def generate_workflow_plan(
        self, 
        goal: str, 
        workflow_inputs: List[Parameter] = [Parameter(name="workflow_input", type="string", description="workflow input")], 
        workflow_outputs: List[Parameter] = [Parameter(name="workflow_output", type="string", description="workflow output")], 
        **kwargs
    ) -> WorkFlowGraph:
        """
        Generates a workflow plan based on the given goal, workflow inputs, and workflow outputs.
        Also validates the workflow plan by checking the workflow graph structure.

        Args:
            goal (str): The goal to generate a workflow for
            workflow_inputs (List[Parameter]): List of workflow inputs
            workflow_outputs (List[Parameter]): List of workflow outputs
            **kwargs: Additional arguments to pass to the task planner agent
        
        Returns:
            WorkFlowGraph: The generated workflow graph
        """

        logger.info("Generating workflow plan...")
        plan = self.generate_plan(goal=goal, workflow_inputs=workflow_inputs, workflow_outputs=workflow_outputs, **kwargs)
        logger.info("Building workflow from plan...")
        workflow = self.build_workflow_from_plan(goal=goal, plan=plan)
        logger.info("Validating initial workflow structure...")
        workflow._validate_workflow_structure()
        logger.info(f"Successfully generate the following workflow:\n{workflow.get_workflow_description()}")
        return workflow


    def generate_plan(
        self, 
        goal: str, 
        workflow_inputs: List[Parameter] = [Parameter(name="workflow_input", type="string", description="workflow input")], 
        workflow_outputs: List[Parameter] = [Parameter(name="workflow_output", type="string", description="workflow output")],
        history: Optional[str] = None, 
        suggestion: Optional[str] = None, 
        examples: Optional[List[Dict]] = None
    ) -> TaskPlanningOutput:
        """
        Generates a workflow plan based on the given goal, workflow inputs, and workflow outputs.
        
        Args:
            goal (str): The goal to generate a workflow plan for
            workflow_inputs (List[Parameter]): List of workflow inputs
            workflow_outputs (List[Parameter]): List of workflow outputs
            history (Optional[str]): Optional history of the conversation
            suggestion (Optional[str]): Optional suggestion for the workflow plan
            examples (Optional[List[Dict]]): Optional examples of similar workflows
        
        Returns:
            TaskPlanningOutput: The generated plan
        """

        task_planner: TaskPlanner = self.task_planner
        task_planning_action_data = {
            "goal": goal, 
            "workflow_inputs": workflow_inputs,
            "workflow_outputs": workflow_outputs,
            "history": history, 
            "suggestion": suggestion, 
            "examples": examples
        }
        task_planning_action_name = task_planner.task_planning_action_name
        message: Message = task_planner.execute(
            action_name=task_planning_action_name,
            action_input_data=task_planning_action_data,
            return_msg_type=MessageType.REQUEST
        )
        return message.content


    def generate_agents(
        self, 
        workflow: WorkFlowGraph,
        examples: Optional[List[Dict]] = None,
        # history: Optional[str] = None, 
        # suggestion: Optional[str] = None
    ) -> WorkFlowGraph:
        """
        Generates agents for a given workflow plan.
        
        Args:
            workflow (WorkFlowGraph): The workflow plan to generate agents for
            examples (Optional[List[Dict]]): Optional examples of similar workflows
        
        Returns:
            WorkFlowGraph: The workflow graph with generated/prebuilt agents
        """
        workflow = asyncio.run(self.async_generate_agents(workflow, examples))
        logger.info("Validating workflow after agent generation...")
        workflow._validate_workflow_structure()
        # Validate that all nodes have agents
        for node in workflow.nodes:
            if not node.agents:
                raise ValueError(f"Node '{node.name}' has no agents assigned after agent generation")
        return workflow


    async def async_generate_agents(
        self, 
        workflow: WorkFlowGraph,
        examples: Optional[List[Dict]] = None,
        # history: Optional[str] = None, 
        # suggestion: Optional[str] = None
    ) -> WorkFlowGraph:

        if examples is not None:
            self._create_agents_rag_engine(examples)

        workflow_desc = workflow.get_workflow_description()
        agent_generation_tasks = [self._generate_agents_for_node(node, workflow.goal, workflow_desc) for node in workflow.nodes]
        processed_nodes = await asyncio.gather(*agent_generation_tasks)
        workflow.nodes = processed_nodes
        return workflow
    

    async def _generate_agents_for_node(self, node: WorkFlowNode, goal: str, workflow_desc: str) -> WorkFlowNode:
        if self.agents_rag_engine is not None:
            rag_results = await self.agents_rag_engine.query_async(node.description)
            chunks = rag_results.corpus.chunks
            agent_examples = [json.loads(chunk.metadata.custom_fields["agent"]) for chunk in chunks]
        else:
            agent_examples = None

        node_fields = ["name", "description", "inputs", "outputs"]
        node_dict = node.to_dict(ignore=["class_name"])
        node_data = {key: value for key, value in node_dict.items() if key in node_fields}
        node_desc = json.dumps(node_data, indent=4)
        agent_generation_action_data = {
            "goal": goal, 
            "workflow": workflow_desc, 
            "task": node_desc, 
            "prebuilt_agents": self.prebuilt_agents,
            "tools": self.tools,
            "examples": agent_examples,
        }
        logger.info(f"Generating agents for node: {node_data['name']}")
        agent_generator_output: AgentGenerationOutput = await self.agent_generator.async_execute(
            action_name=self.agent_generator.agent_generation_action_name, 
            action_input_data=agent_generation_action_data,
            return_msg_type=MessageType.RESPONSE
        )
        
        agents = self._process_agent_generator_output(agent_generator_output.content)
        node.set_agents(agents=agents)
        return node


    # def review_plan(self, goal: str, )
    def build_workflow_from_plan(self, goal: str, plan: TaskPlanningOutput) -> WorkFlowGraph:
        """
        Builds a workflow graph from task planner agent's output by setting sub-tasks as nodes
        and infer edges between nodes based on their inputs and outputs.
        
        Args:
            goal (str): The goal of the workflow
            plan (TaskPlanningOutput): The task planner agent's output
        
        Returns:
            WorkFlowGraph: The workflow graph
        """
        nodes: List[WorkFlowNode] = plan.sub_tasks
        # infer edges from sub-tasks' inputs and outputs
        edges: List[WorkFlowEdge] = []
        for node in nodes:
            for another_node in nodes:
                if node.name == another_node.name:
                    continue
                node_output_params = [param.name for param in node.outputs]
                another_node_input_params = [param.name for param in another_node.inputs]
                if any([param in another_node_input_params for param in node_output_params]):
                    edges.append(WorkFlowEdge(edge_tuple=(node.name, another_node.name)))
        workflow = WorkFlowGraph(goal=goal, nodes=nodes, edges=edges)
        return workflow


    def _get_workflow_examples(self, goal: str) -> List[Dict]:
        rag_result: List[Chunk] = self.rag_engine.query(query=goal).corpus.chunks
        if len(rag_result) == 0:
            logger.warning("No relevant workflow examples found.")
            return []
        
        workflow_examples: List[Dict] = []

        for chunk in rag_result:
            file_path = os.path.join(self.workflow_folder, chunk.metadata.file_name)
            with open(file_path, "r") as f:
                workflow_examples.append(json.load(f))

        return workflow_examples


    @staticmethod
    def _extract_tasks_and_agents(workflow_dict: Dict) -> tuple[Dict, List[Dict]]:
        """
        Extracts tasks and agents from a workflow dictionary.
        
        Args:
            workflow_dict (Dict): The workflow dictionary
        
        Returns:
            tuple[Dict, List[Dict]]: A tuple containing the tasks and agents

        tasks example:
            {
                "goal": "...",
                "sub_tasks": [
                    {
                        "name": "...",
                        "description": "...",
                        "inputs": [
                            {
                                "name": "...",
                                "description": "...",
                                "type": "..."
                            }
                        ],
                        "outputs": [
                            {
                                "name": "...",
                                "description": "...",
                                "type": "..."
                            }
                        ]
                    },
                    ...
                ]
            }

        agents example:
            [
                {
                    "subtask": {
                        ...
                    },
                    "agents": [
                        {
                            "name": "...",
                            "description": "...",
                            "inputs": [
                                {
                                    "name": "...",
                                    "description": "...",
                                    "type": "..."
                                }
                            ],
                            "outputs": [
                                {
                                    "name": "...",
                                    "description": "...",
                                    "type": "..."
                                }
                            ],
                            "prompt_template": {
                                "class_name": "ChatTemplate",
                                "instruction": "..."
                            },
                            "tool_names": [...]
                        },
                        ...
                    ]
                },
                ...
            ]
        """
        keys_to_remove = [
            "class_name",
            "required",
            "status"
        ]
        nodes = workflow_dict["nodes"]
        tasks = []
        agents = []

        for node in nodes:
            node_agents = node.pop("agents")
            formatted_node_agents = []

            for node_agent in node_agents:
                agent = {
                    "name": node_agent["name"],
                    "description": node_agent["description"],
                    "inputs": node_agent["inputs"],
                    "outputs": node_agent["outputs"],
                    "prompt_template": node_agent["prompt_template"],
                    "tool_names": node_agent.get("tool_names", None)
                }
                formatted_node_agents.append(agent)

            subtask = recursive_remove(node, keys_to_remove)

            # keep "reason" in task planning examples
            # but remove "reason" in agent examples
            agent_subtask = deepcopy(subtask)
            agent_subtask.pop("reason", None)
            agents.append({
                "subtask": agent_subtask,
                "agents": formatted_node_agents
            })
            tasks.append(subtask)

        tasks = {"goal": workflow_dict["goal"], "sub_tasks": tasks}
        return tasks, agents


    def _get_examples(self, goal: str) -> Tuple[Optional[List[Dict]], Optional[List[Dict]]]:
        """Get examples for task planning and agent generation.
        
        Args:
            goal (str): Examples that share a similar goal are retrieved.
        
        Returns:
            A tuple that contains:
                - Examples for task planning.
                - Examples for agent generation.
        """
        workflow_examples = self._get_workflow_examples(goal)
        if len(workflow_examples) == 0:
            return None, None
                
        task_planning_examples = []
        agent_examples = []
        for workflow in workflow_examples:
            task_planning_example, agent_example = self._extract_tasks_and_agents(workflow)
            task_planning_examples.append(task_planning_example)
            agent_examples.extend(agent_example)
        return task_planning_examples, agent_examples


    def _create_agents_rag_engine(self, data: List[Dict]):
        """Create a temporary RAGEngine to retrieve examples for agent generation.
        
        Args:
            data (List[Dict]): A list of dictionaries, each should contains a "subtask" key and an "agents" key
        """

        store_config = StoreConfig(
            dbConfig=DBConfig(),
            vectorConfig=VectorStoreConfig(),
        )
        storage_handler = StorageHandler(storageConfig=store_config)

        self.agents_rag_engine = RAGEngine(config=self.rag_engine.config, storage_handler=storage_handler)

        chunks = []
        for item in data:
            task_description = item["subtask"]["description"]
            metadata = ChunkMetadata(custom_fields={"agent": json.dumps(item)})
            chunks.append(Chunk(text=task_description, metadata=metadata))

        self.agents_rag_engine.add(index_type="vector", nodes=Corpus(chunks=chunks))


    def _process_agent_generator_output(self, output: AgentGenerationOutput) -> List[Union[Dict, AgentAdaptor]]:
        """Converts `AgentGenerationOutput` to list of dict or `AgentAdaptor`.
        The dicts can be used to construct a `CustomizeAgent`. The `AgentAdaptor`
        instances correspond to the selected prebuilt agents.
        """
        selected_agents = self._processs_selected_agents(output.selected_agents)
        generated_agents = self._process_generated_agents(output.generated_agents)
        return [*selected_agents, *generated_agents]


    def _processs_selected_agents(self, agents: List[SelectedAgent]) -> List[Agent]:
        """Converts a list of `SelectedAgent` to `AgentAdaptor` and returns their dict representation"""
        selected_agents = []

        if len(agents) == 0:
            return []
        
        for agent in agents:
            agent = self.selected_agents_to_agents(agent)
            selected_agents.append(agent)
        return selected_agents


    def selected_agents_to_agents(self, agent: SelectedAgent) -> Agent:
        """Converts `SelectedAgent` to `AgentAdaptor`"""
        prebuilt_agent = self.prebuilt_agents_map[agent.name]

        # Creates unique agent adaptor name
        agent_adaptor_name = agent.name + "V1"
        i = 2
        while agent_adaptor_name in self.agent_adaptor_names:
            agent_adaptor_name = f"{agent.name}V{i}"
            i += 1
        self.agent_adaptor_names.append(agent_adaptor_name)

        agent_adaptor = AgentAdaptor(
            name=agent_adaptor_name,
            agent=prebuilt_agent,
            inputs=agent.inputs,
            outputs=agent.outputs
        )
        return agent_adaptor
    

    @staticmethod
    def _process_generated_agents(agents: List[GeneratedAgent]) -> List[Dict]:
        """
        Converts `GeneratedAgent` to dict and removes class_name from agent's dict, inputs and outputs
        """
        generated_agents = []
        for agent in agents:
            agent_dict = agent.to_dict()
            agent_dict.pop("class_name")
            generated_agents.append(agent_dict)
        return generated_agents


    def _prepare_rag_engine(self):
        logger.info("Preparing RAG engine...")

        if self.rag_engine is not None:
            return

        db_path = self.db_path if self.db_path is not None else "./workflows.db"

        storage_handler = StorageHandler(
            storageConfig=StoreConfig(
                dbConfig=DBConfig(db_name="sqlite", path=db_path),
                vectorConfig=VectorStoreConfig()
            )
        )
        rag_config = RAGConfig(
            retrieval=RetrievalConfig(top_k=3, similarity_cutoff=0),
            embedding=EmbeddingConfig(
                provider="openai",
                model_name="text-embedding-ada-002"
            )
        )
        self.rag_engine = RAGEngine(config=rag_config, storage_handler=storage_handler)

        if self.db_path is not None:
            self.rag_engine.load()
            return

        if not os.path.exists(self.workflow_folder):
            logger.warning(f"Cannot find workflow folder at '{self.workflow_folder}'. RAG will not be used.")
            self.rag_engine = None
            return

        chunks = []
        for file in os.listdir(self.workflow_folder):
            if file.endswith(".json"):
                workflow_dict = json.load(open(os.path.join(self.workflow_folder, file)))
                metadata = ChunkMetadata(file_name=file)
                workflow_goal = workflow_dict.get("goal", None)

                if workflow_goal is None:
                    logger.warning(f"Invalid workflow json file {file}. Missing 'goal' key. Skipping.")
                    continue
                    
                chunk = Chunk(
                    text=workflow_dict["goal"],
                    metadata=metadata
                )
                chunks.append(chunk)
        
        self.rag_engine.add(index_type="vector", nodes=Corpus(chunks=chunks))
        if len(chunks) == 0:
            logger.warning("No workflow json files found in the workflow folder. RAG will not be used.")
            self.rag_engine = None
        else:
            logger.info(f"Successfully added {len(chunks)} workflows to the RAG engine.")
            self.rag_engine.save()

