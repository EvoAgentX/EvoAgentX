import os
import random  
from typing import Any, List

from .benchmark import Benchmark
from .measures import exact_match_score
from ..core.logging import logger
from ..core.module_utils import load_json
from ..utils.utils import download_file

MULTIPLE_CHOICE_TASKS = [
    'temporal_sequences', 'disambiguation_qa', 'date_understanding', 'tracking_shuffled_objects_three_objects', 'penguins_in_a_table', 
    'geometric_shapes', 'snarks', 'ruin_names', 'tracking_shuffled_objects_seven_objects', 'tracking_shuffled_objects_five_objects', 
    'logical_deduction_three_objects', 'hyperbaton', 'logical_deduction_five_objects', 'logical_deduction_seven_objects', 'movie_recommendation', 
    'salient_translation_error_detection', 'reasoning_about_colored_objects', 
]
FREE_FORM_TASKS = [
    'multistep_arithmetic_two', 'navigate', 'dyck_languages', 'word_sorting', 'sports_understanding', 
    'boolean_expressions', 'object_counting', 'formal_fallacies', 'causal_judgement', 'web_of_lies', 
]
ALL_TASKS = {task: f"{task}.json" for task in MULTIPLE_CHOICE_TASKS + FREE_FORM_TASKS}

def download_raw_bigbenchhard_data(task_name: str, save_folder: str):
    assert task_name in ALL_TASKS, f"'{task_name}' is an invalid bigbenchhard task name. Available tasks: {list(ALL_TASKS.keys())}"
    file_name = ALL_TASKS[task_name]
    url = f"https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/bbh/{file_name}"
    logger.info(f"Downloading BIGBenchHard '{task_name}' data from: {url}")
    download_file(url=url, save_file=os.path.join(save_folder, file_name))


class BIGBenchHard(Benchmark):

    def __init__(self, task: str, path: str = None, sample_num: int = 0, **kwargs):
        path = os.path.expanduser(path or f"~/.evoagentx/data/bigbenchhard/{task}")
        
        if task not in ALL_TASKS:
            raise ValueError(f"Unknown task '{task}'. Available tasks: {list(ALL_TASKS.keys())}")
            
        self.task = task
        self.file_name = ALL_TASKS[task]
        self.sample_num = sample_num  
        
        super().__init__(name=f"BIGBenchHard-{self.task}", path=path, **kwargs)

    def _load_data_from_file(self, file_name: str):
        if file_name is None:
            return None
        file_path = os.path.join(self.path, file_name)
        if not os.path.exists(file_path):
            download_raw_bigbenchhard_data(task_name=self.task, save_folder=self.path)
        logger.info(f"Loading BIGBenchHard data from {file_path} ...")
        data = load_json(path=file_path, type="json")
        return data.get("examples", [])

    def _load_data(self):

        task_data = self._load_data_from_file(file_name=self.file_name)
        
        if self.sample_num > 0 and len(task_data) > self.sample_num:
            logger.info(f"Randomly sampling {self.sample_num} examples for the dev set.")
            self._dev_data = random.sample(task_data, self.sample_num)
            

            dev_set_tuples = {tuple(d.items()) for d in self._dev_data}
            self._test_data = [item for item in task_data if tuple(item.items()) not in dev_set_tuples]
        else:
            if self.sample_num > 0:
                logger.info(f"Warning: sample_num ({self.sample_num}) is >= total data size ({len(task_data)}). Using all data for dev set and none for test set.")
                self._dev_data = task_data
                self._test_data = []
            else:
                logger.info("sample_num is 0, using all data for the test set.")
                self._dev_data = []
                self._test_data = task_data

        self._data = task_data
        self._train_data = [] 
    def get_input_keys(self) -> List[str]:
        return ["input"]
    
    def _get_label(self, example: Any) -> Any:
        return example["target"]
    
    def _get_id(self, example: Any) -> Any:
        return None 
    
    def evaluate(self, prediction: Any, label: Any) -> dict:
        em = exact_match_score(prediction=prediction, ground_truth=label)
        return {"em": em}