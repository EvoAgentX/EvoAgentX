# Acknowledgement: Modified from AFlow (https://github.com/geekan/MetaGPT/blob/main/metagpt/ext/aflow/scripts/optimizer_utils/graph_utils.py) under MIT License 

import os 
import re
import time
import traceback
from typing import List
from pathlib import Path
from ...core.logging import logger
from ...prompts.optimizers.aflow_optimizer import (
    WORKFLOW_INPUT, 
    WORKFLOW_OPTIMIZE_PROMPT, 
    WORKFLOW_CUSTOM_USE, 
    WORKFLOW_TEMPLATE
)
from ...models.base_model import BaseLLM 
from ...workflow.operators import (
    Operator, Custom, CustomCodeGenerate, 
    ScEnsemble, Test, AnswerGenerate, QAScEnsemble, Programmer
)

OPERATOR_MAP = {
    "Custom": Custom,
    "CustomCodeGenerate": CustomCodeGenerate,
    "ScEnsemble": ScEnsemble,
    "Test": Test,
    "AnswerGenerate": AnswerGenerate,
    "QAScEnsemble": QAScEnsemble,
    "Programmer": Programmer
}


class GraphUtils:

    def __init__(self, root_path: str):
        self.root_path = root_path

    def create_round_directory(self, graph_path: str, round_number: int) -> str:
        directory = os.path.join(graph_path, f"round_{round_number}")
        os.makedirs(directory, exist_ok=True)
        return directory

    def load_graph(self, round_number: int, workflows_path: str):
        workflows_path = workflows_path.replace("\\", ".").replace("/", ".")
        graph_module_name = f"{workflows_path}.round_{round_number}.graph"
        try:
            graph_module = __import__(graph_module_name, fromlist=[""])
            graph_class = getattr(graph_module, "Workflow")
            return graph_class
        except ImportError as e:
            logger.info(f"Error loading graph for round {round_number}: {e}")
            raise

    def read_graph_files(self, round_number: int, workflows_path: str):
        prompt_file_path = os.path.join(workflows_path, f"round_{round_number}", "prompt.py")
        graph_file_path = os.path.join(workflows_path, f"round_{round_number}", "graph.py")

        try:
            with open(prompt_file_path, "r", encoding="utf-8") as file:
                prompt_content = file.read()
            with open(graph_file_path, "r", encoding="utf-8") as file:
                graph_content = file.read()
        except FileNotFoundError as e:
            logger.info(f"Error: File not found for round {round_number}: {e}")
            raise
        except Exception as e:
            logger.info(f"Error loading prompt for round {round_number}: {e}")
            raise
        return prompt_content, graph_content

    def extract_solve_graph(self, graph_load: str) -> List[str]:
        pattern = r"class Workflow:.+"
        return re.findall(pattern, graph_load, re.DOTALL)

    def load_operators_description(self, operators: List[str], llm: BaseLLM) -> str:

        operators_description = ""
        for id, operator in enumerate(operators):
            operator_description = self._load_operator_description(id + 1, operator, llm)
            operators_description += f"{operator_description}\n"
        return operators_description

    def _load_operator_description(self, id: int, operator_name: str, llm: BaseLLM) -> str:
        if operator_name not in OPERATOR_MAP:
            raise ValueError(f"Operator {operator_name} not Found in OPERATOR_MAP! Available operators: {OPERATOR_MAP.keys()}")
        operator: Operator = OPERATOR_MAP[operator_name](llm=llm)
        return f"{id}. {operator_name}: {operator.description}, with interface {operator.interface})."

    def create_graph_optimize_prompt(
        self,
        experience: str,
        score: float,
        graph: str,
        prompt: str,
        operator_description: str,
        type: str,
        log_data: str,
    ) -> str:
        graph_input = WORKFLOW_INPUT.format(
            experience=experience,
            score=score,
            graph=graph,
            prompt=prompt,
            operator_description=operator_description,
            type=type,
            log=log_data,
        )
        graph_system = WORKFLOW_OPTIMIZE_PROMPT.format(type=type)
        return graph_input + WORKFLOW_CUSTOM_USE + graph_system

    def get_graph_optimize_response(self, graph_optimize_node):
        max_retries = 5
        retries = 0

        while retries < max_retries:
            try:
                response = graph_optimize_node.instruct_content.model_dump()
                return response
            except Exception as e:
                retries += 1
                logger.info(f"Error generating prediction: {e}. Retrying... ({retries}/{max_retries})")
                if retries == max_retries:
                    logger.info("Maximum retries reached. Skipping this sample.")
                    break
                traceback.print_exc()
                time.sleep(5)
        return None

    def write_graph_files(self, directory: str, response: dict):
        
        graph = WORKFLOW_TEMPLATE.format(graph=response["graph"])        
        with open(os.path.join(directory, "graph.py"), "w", encoding="utf-8") as file:
            file.write(graph)
        with open(os.path.join(directory, "prompt.py"), "w", encoding="utf-8") as file:
            prompt = response["prompt"].replace("prompt_custom.", "")
            file.write(prompt)
        with open(os.path.join(directory, "__init__.py"), "w", encoding="utf-8") as file:
            file.write("")
        self.update_prompt_import(os.path.join(directory, "graph.py"), directory)
    
    def update_prompt_import(self, graph_file: str, prompt_folder: str):

        project_root = Path(os.getcwd())
        prompt_folder_path = Path(prompt_folder)

        if not prompt_folder_path.is_absolute():
            prompt_folder_full_path = Path(os.path.join(project_root, prompt_folder))
            if not prompt_folder_full_path.exists():
                raise ValueError(f"Prompt folder {prompt_folder_full_path} does not exist!")
            prompt_folder_path = prompt_folder_full_path
        
        try:
            relative_path = prompt_folder_path.relative_to(project_root)
        except ValueError:
            raise ValueError(f"Prompt folder {prompt_folder} must be within the project directory")

        import_path = str(relative_path).replace(os.sep, ".")
        if import_path.startswith("."):
            import_path = import_path[1:]
        
        with open(graph_file, "r", encoding="utf-8") as file:
            graph_content = file.read()

        # 在graph_content中找到import语句
        pattern = r'import .*?\.prompt as prompt_custom' 
        replacement = f'import {import_path}.prompt as prompt_custom'
        new_content = re.sub(pattern, replacement, graph_content)

        with open(graph_file, "w", encoding="utf-8") as file:
            file.write(new_content)
        

        