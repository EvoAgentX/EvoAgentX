import os 
import tarfile
from ..utils import download_file
from ...core.logging import logger
from ...prompts.aflow_optimize_prompt import WORKFLOW_INPUT, WORKFLOW_OPTIMIZE_PROMPT, WORKFLOW_CUSTOM_USE, WORKFLOW_TEMPLATE
from typing import List
import re
import json
import time
import traceback

AFLOW_DATASET_FILES_MAP = {
    "hotpotqa": {"train": None, "dev": "hotpotqa_validate.jsonl", "test": "hotpotqa_test.jsonl"},
    "humaneval": {"train": None, "dev": "humaneval_validate.jsonl", "test": "humaneval_test.jsonl"},
}

def extract_tar_gz(filename: str, extract_path: str) -> None:
    """Extract a tar.gz file to the specified path."""
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=extract_path)


def download_aflow_benchmark_data(dataset: str, save_folder: str):
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    candidate_datasets = list(AFLOW_DATASET_FILES_MAP.keys()) + ["all"]
    lower_candidate_datasets = [dataset.lower() for dataset in candidate_datasets]
    if dataset.lower() not in lower_candidate_datasets:
        raise ValueError(f"Invalid value for dataset: {dataset}. Available choices: {candidate_datasets}")
    
    url = "https://drive.google.com/uc?export=download&id=1DNoegtZiUhWtvkd2xoIuElmIi4ah7k8e"
    logger.info(f"Downloading AFlow benchmark data from {url} ...")
    aflow_data_save_file = os.path.join(save_folder, "aflow_data.tar.gz")
    download_file(url=url, save_file=aflow_data_save_file)

    logger.info(f"Extracting data for {dataset} dataset(s) from {aflow_data_save_file} ...")
    extract_tar_gz(aflow_data_save_file, extract_path=save_folder)

    if dataset != "all":
        dataset_files = [file for file in list(AFLOW_DATASET_FILES_MAP[dataset].values()) if file is not None]
        for file in os.listdir(save_folder):
            if file not in dataset_files:
                os.remove(os.path.join(save_folder, file))
    
    if os.path.exists(aflow_data_save_file):
        logger.info(f"Remove {aflow_data_save_file}")
        os.remove(aflow_data_save_file)
        
        
class GraphUtils:
    def __init__(self, root_path: str):
        self.root_path = root_path

    def create_round_directory(self, graph_path: str, round_number: int) -> str:
        directory = os.path.join(graph_path, f"round_{round_number}")
        os.makedirs(directory, exist_ok=True)
        return directory

    def load_graph(self, round_number: int, workflows_path: str):
        workflows_path = workflows_path.replace("\\", ".").replace("/", ".")
        # logger.info(f"Loading graph for round {round_number} from {workflows_path} ...")
        graph_module_name = f"{workflows_path}.round_{round_number}.graph"
        # logger.info(f"graph_module_name is {graph_module_name}")
        # breakpoint()
        try:
            graph_module = __import__(graph_module_name, fromlist=[""])
            # logger.info(f"graph_module is {graph_module}")
            graph_class = getattr(graph_module, "AFlowWorkflowGenerator")
            # logger.info(f"graph_class is {graph_class}")
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
        pattern = r"class AFlowWorkflowGenerator(?:\(WorkFlowGenerator\))?:.+"
        return re.findall(pattern, graph_load, re.DOTALL)

    def load_operators_description(self, operators: List[str]) -> str:
        # path = f"{self.root_path}/workflows/template/operator.json"
        # path = f"../workflow/operator.json"
        path = "evoagentx/workflow/operator.json"
        operators_description = ""
        for id, operator in enumerate(operators):
            operator_description = self._load_operator_description(id + 1, operator, path)
            operators_description += f"{operator_description}\n"
        return operators_description

    def _load_operator_description(self, id: int, operator_name: str, file_path: str) -> str:
        with open(file_path, "r") as f:
            operator_data = json.load(f)
            matched_data = operator_data[operator_name]
            desc = matched_data["description"]
            interface = matched_data["interface"]
            return f"{id}. {operator_name}: {desc}, with interface {interface})."

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
        
        # logger.info(f"graph_input is {graph_input}")
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

    def write_graph_files(self, directory: str, response: dict, round_number: int, dataset: str):
        graph = WORKFLOW_TEMPLATE.format(graph=response["graph"], round=round_number, dataset=dataset)

        # logger.info(f"graph is {graph}")
        
        
        with open(os.path.join(directory, "graph.py"), "w", encoding="utf-8") as file:
            file.write(graph)

        with open(os.path.join(directory, "prompt.py"), "w", encoding="utf-8") as file:
            file.write(response["prompt"])

        with open(os.path.join(directory, "__init__.py"), "w", encoding="utf-8") as file:
            file.write("")
