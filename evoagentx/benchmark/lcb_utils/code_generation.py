# copied from: https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/benchmarks/code_generation.py

import json
import zlib
import pickle
import base64
from enum import Enum
from datetime import datetime
from dataclasses import dataclass

from datasets import load_dataset


class Platform(Enum):
    """Enum representing the coding platforms supported in the dataset.
    
    Defines the platforms where programming problems are sourced from.
    
    Attributes:
        LEETCODE: LeetCode platform
        CODEFORCES: Codeforces platform
        ATCODER: AtCoder platform
    """
    LEETCODE = "leetcode"
    CODEFORCES = "codeforces"
    ATCODER = "atcoder"


class Difficulty(Enum):
    """Enum representing the difficulty levels of coding problems.
    
    Standard difficulty categorization for programming problems.
    
    Attributes:
        EASY: Entry-level problems
        MEDIUM: Intermediate difficulty problems
        HARD: Advanced difficulty problems
    """
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class TestType(Enum):
    """Enum representing the types of test cases.
    
    Defines how inputs and outputs for test cases should be processed.
    
    Attributes:
        STDIN: Test cases where input is provided via standard input
        FUNCTIONAL: Test cases where input is provided as function arguments
    """
    STDIN = "stdin"
    FUNCTIONAL = "functional"


@dataclass
class Test:
    """Represents a test case for code evaluation.
    
    Contains input, expected output, and the type of test (stdin or functional).
    
    Attributes:
        input: The input data for the test case
        output: The expected output data for the test case
        testtype: The type of test (STDIN or FUNCTIONAL)
    """
    input: str
    output: str
    testtype: TestType

    def __post_init__(self):
        """Post-initialization processing.
        
        Converts the string test type to a TestType enum value.
        """
        self.testtype = TestType(self.testtype)
        # if self.testtype == TestType.FUNCTIONAL:
        #     self.input = json.loads(self.input)
        #     self.output = json.loads(self.output)


@dataclass
class CodeGenerationProblem:
    """Represents a code generation problem from competitive programming platforms.
    
    This class encapsulates all the information related to a programming problem,
    including problem statement, test cases, metadata, and evaluation methods.
    
    Attributes:
        question_title: Title of the programming problem
        question_content: Detailed problem statement
        platform: The platform the problem is from (LeetCode, Codeforces, AtCoder)
        question_id: Unique identifier for the question on its platform
        contest_id: Identifier for the programming contest
        contest_date: Date when the contest was held
        starter_code: Initial code template provided to the user
        difficulty: Problem difficulty level (easy, medium, hard)
        public_test_cases: List of test cases visible to the contestant
        private_test_cases: List of hidden test cases used for evaluation
        metadata: Additional information about the problem
    """
    question_title: str
    question_content: str
    platform: Platform
    question_id: str
    contest_id: str
    contest_date: datetime
    starter_code: str
    difficulty: Difficulty
    public_test_cases: list[Test]
    private_test_cases: list[Test]
    metadata: dict

    def __post_init__(self):
        """Post-initialization processing.
        
        Converts string values to appropriate types and decodes compressed test cases.
        This includes:
        1. Converting platform and difficulty strings to enum values
        2. Parsing the contest date string
        3. Deserializing public and private test cases
        4. Parsing metadata
        """
        self.platform = Platform(self.platform)
        self.difficulty = Difficulty(self.difficulty)
        self.contest_date = datetime.fromisoformat(self.contest_date)

        self.public_test_cases = json.loads(self.public_test_cases)  # type: ignore
        self.public_test_cases = [Test(**t) for t in self.public_test_cases]

        try:
            self.private_test_cases = json.loads(self.private_test_cases)  # type: ignore
        except Exception:
            self.private_test_cases = json.loads(
                pickle.loads(
                    zlib.decompress(
                        base64.b64decode(self.private_test_cases.encode("utf-8"))  # type: ignore
                    )
                )
            )  # type: ignore
        self.private_test_cases = [Test(**t) for t in self.private_test_cases]

        self.metadata = json.loads(self.metadata)  # type: ignore

    def insert_output(self, output_list: list[str], code_list: list[str]) -> dict:
        """Create a result dictionary with model outputs and generated code.
        
        Args:
            output_list: List of raw outputs from the model
            code_list: List of code solutions generated by the model
            
        Returns:
            Dictionary containing problem data along with model outputs and generated code
        """
        return {
            "question_title": self.question_title,
            "question_content": self.question_content,
            "platform": self.platform.value,
            "question_id": self.question_id,
            "contest_id": self.contest_id,
            "contest_date": self.contest_date.isoformat(),
            "starter_code": self.starter_code,
            "difficulty": self.difficulty.value,
            "output_list": output_list,
            "code_list": code_list,
        }

    def insert_output_evaluation(
        self,
        output_list: list[str],
        code_list: list[str],
        graded_list: list[bool],
        **kwargs,
    ) -> dict:
        """Create a result dictionary with model outputs and evaluation results.
        
        Extends the insert_output method by adding evaluation metrics, including
        the pass@1 score which represents the success rate of the model.
        
        Args:
            output_list: List of raw outputs from the model
            code_list: List of code solutions generated by the model
            graded_list: List of boolean values indicating whether each solution passed
            **kwargs: Additional key-value pairs to include in the result dictionary
            
        Returns:
            Dictionary containing problem data, model outputs, and evaluation metrics
        """
        output = self.insert_output(output_list, code_list)
        output["graded_list"] = graded_list
        output["pass@1"] = graded_list.count(True) / len(graded_list)
        for k, v in kwargs.items():
            output[k] = v
        return output

    def get_evaluation_sample(self):
        """Get a sample for evaluation purposes.
        
        Creates a dictionary containing all test inputs and outputs in JSON format,
        along with the function name if available in metadata.
        
        Returns:
            Dictionary with input_output field containing serialized test data
        """
        return {
            "input_output": json.dumps(
                {
                    "inputs": [
                        t.input
                        for t in self.public_test_cases + self.private_test_cases
                    ],
                    "outputs": [
                        t.output
                        for t in self.public_test_cases + self.private_test_cases
                    ],
                    "fn_name": self.metadata.get("func_name", None),
                }
            ),
        }


def load_code_generation_dataset(release_version="release_v1", cache_dir: str = None, start_date=None, end_date=None) -> list[CodeGenerationProblem]:
    """Load the LiveCodeBench code generation dataset.
    
    Retrieves the code generation problems from the LiveCodeBench dataset using the
    Hugging Face datasets library and optionally filters by date range.
    
    Args:
        release_version: Version of the dataset to load (default: "release_v1")
        cache_dir: Directory to cache the dataset (default: None, uses Hugging Face default)
        start_date: Optional start date to filter problems (format: "YYYY-MM-DD")
        end_date: Optional end date to filter problems (format: "YYYY-MM-DD")
        
    Returns:
        List of CodeGenerationProblem objects representing the filtered dataset
    """
    dataset = load_dataset("livecodebench/code_generation_lite", split="test", version_tag=release_version, trust_remote_code=True, cache_dir=cache_dir)
    dataset = [CodeGenerationProblem(**p) for p in dataset]  # type: ignore
    if start_date is not None:
        p_start_date = datetime.strptime(start_date, "%Y-%m-%d")
        dataset = [e for e in dataset if p_start_date <= e.contest_date]

    if end_date is not None:
        p_end_date = datetime.strptime(end_date, "%Y-%m-%d")
        dataset = [e for e in dataset if e.contest_date <= p_end_date]
    
    # print(f"Loaded {len(dataset)} problems")
    return dataset


def load_code_generation_dataset_not_fast(release_version="release_v1") -> list[CodeGenerationProblem]:
    """Load the full (non-lite) LiveCodeBench code generation dataset.
    
    Retrieves the full code generation dataset, which may be larger and slower
    to load than the lite version.
    
    Args:
        release_version: Version of the dataset to load (default: "release_v1")
        
    Returns:
        List of CodeGenerationProblem objects representing the dataset
    """
    dataset = load_dataset("livecodebench/code_generation", split="test")
    dataset = [CodeGenerationProblem(**p) for p in dataset]  # type: ignore
    # print(f"Loaded {len(dataset)} problems")
    return dataset


if __name__ == "__main__":
    dataset = load_code_generation_dataset()