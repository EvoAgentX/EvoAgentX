import os 
import tarfile
from .utils import download_file
from ..core.logging import logger

AFLOW_DATASET_FILES_MAP = {
    "hotpotqa": {"train": None, "dev": "hotpotqa_validate.jsonl", "test": "hotpotqa_test.jsonl"},
}

def extract_tar_gz(filename: str, extract_path: str) -> None:
    """Extract a tar.gz file to the specified path."""
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=extract_path)


def download_aflow_benchmark_data(dataset: str, save_folder: str):

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