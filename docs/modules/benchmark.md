# Benchmark 

## Benchmark Overview 

EvoAgentX provides a set of benchmarks to facilitate the evaluation of different agent-based systems. Below is a summary of the benchmarks currently included, along with basic dataset statistics: 


| Task                      | Dataset Name    | # Train   | # Dev   | # Test |
| ------------------------- | --------------- | --------- | ------- | ------ |
| QA                        | NQ              | 79,168    | 8,757   | 3,610  |
| Multi-Hop QA              | HotPotQA        | 90,447    | 7,405   | /      |
| Math                      | GSM8K           | 7,473     | /       | 1,319  |
| Math                      | MATH            | 7,500     | /       | 5,000  |
| Code Generation           | HumanEval       | /         | /       | 164    |
| Code Generation           | MBPP            | /         | /       | 427    |
| Code Generation           | LiveCodeBench(v1~v5) | /         | /       | 400~880    |
| Code Execution            | LiveCodeBench   | /         | /       | 479      |
| Test Output Prediction    | LiveCodeBench   | /         | /       | 442      |

Our framework provides automatic dataset downloading capabilities, and all benchmarks have built-in evaluation methods. The framework is designed to allow users to easily load, use, and evaluate datasets for various tasks without manually handling data downloading and evaluation logic. 
All datasets are automatically downloaded to the default path (~/.evoagentx/data/) when first used, or users can specify a custom path via parameter `path`. Each benchmark class implements a standardized interface, including methods for data loading, label retrieval, and prediction evaluation. 

Below, we introduce the preprocessing steps and evaluation metrics for each benchmark. 

- [Question Answering](#question-answering)
    - [NQ](#nq)
    - [HotPotQA](#hotpotqa)
- [Math](#math)
    - [GSM8K](#gsm8k)
    - [MATH](#math)
- [Code Generation](#code-generation)
    - [HumanEval](#humaneval)
    - [MBPP](#mbpp)
    - [LiveCodeBench](#livecodebench)

## Preprocessing and Evaluation Metrics 

### Question Answering 

For the QA datasets, we use Exact Match (EM), F1, and Accuracy (ACC) as evaluation metrics by default. EM requires the predicted answer to be exactly the same as the ground truth answer. ACC requires that the predicted answer contains the ground-truth answer, which is useful when the LLM is used to generate the answer. 

#### NQ
[Natural Questions (NQ)](https://github.com/google-research-datasets/natural-questions) contains questions from the Google search engine and the answers, annotated by human annotators, are paragraphs or entities in the Wikipedia page of the top 5 search results. We use the dataset splits provided by the [DPR](https://github.com/facebookresearch/DPR) repository, which contains 79,168 training, 8,757 development, and 3,610 test examples. 

You can load the dataset using the following code: 
```python
from evoagentx.benchmark import NQ
nq_dataset = NQ() # optional: path="/path/to/save_data"
test_data = nq_dataset.get_test_data()
```
Each example in the dataset is in the following format: 
```json
{
    "id": "test-1", 
    "question": "the question", 
    "answers": ["possible answers"]
}
```


#### HotPotQA 
[HotPotQA](https://hotpotqa.github.io/) is a multi-hop QA dataset that requires multi-step reasoning to answer the question. We use the distractor setting of the dataset. Each example contains a question, an answer, some context that contians both supporting and distractor information, and supporting facts. We only include the training and development sets, as the test set is not publicly available. 

You can load the dataset using the following code: 
```python
from evoagentx.benchmark import HotPotQA
hotpotqa_dataset = HotPotQA() # optional: path="/path/to/save_data"
test_data = hotpotqa_dataset.get_test_data()
```
Each example in the dataset is in the following format, where the second element (int) of a supporting_fact is the index of the sentence in the context that supports the answer. 
```json
{
        "_id": "the id of the example", 
        "question": "the question", 
        "answer": "the answer", 
        "context": [["context_title", ["context_sentence", "another_sentence"]]],
        "supporting_facts": [["supporting_title", 0]]
    }
```


### Math

For match datasets, we use the solve rate as the evaluation metric. The solve rate is the ratio of the number of examples that are solved correctly to the total number of examples.

#### GSM8K 
[GSM8K](https://github.com/openai/grade-school-math) consists of high quality grade school math problems created by human problem writers. These problems require multi-step mathematical reasoning to solve. We use the dataset splits provided by the original repository, which contains 7.5K training problems and 1K test problems. 

You can load the dataset using the following code: 
```python
from evoagentx.benchmark import GSM8K
gsm8k_dataset = GSM8K() # optional: path="/path/to/save_data"
test_data = gsm8k_dataset.get_test_data()
```
Each example in the dataset is in the following format: 
```json
{
    "id": "test-1", 
    "question": "the question", 
    "answer": "the answer"
}
```

#### MATH 
The [Mathematics Aptitude Test of Heuristics (MATH)](https://github.com/hendrycks/math) dataset consists of problems from mathematics competitions, including the AMC 10, AMC 12, AIME, etc. Each problem in MATH has a step-by-step solution. We use the dataset splits provided by the original repository, which contains 7.5K training problems and 5K test problems. 

You can load the dataset using the following code: 
```python
from evoagentx.benchmark import MATH
math_dataset = MATH() # optional: path="/path/to/save_data"
test_data = math_dataset.get_test_data()
```
Each example in the dataset is in the following format. For the `level` field, valid values are: "Level 1", "Level 2", "Level 3", "Level 4", "Level 5", and "Level ?". The `type` field can be one of: "Geometry", "Algebra", "Intermediate Algebra", "Counting & Probability", "Precalculus", "Number Theory", or "Prealgebra".

```json
{
    "id": "test-1", 
    "problem": "the problem", 
    "solution": "the solution",
    "level": "Level 1",
    "type": "Algebra"
}
```

### Code Generation 
For the code generation benchmarks, we use pass@k as the evaluation metric, where k is the number of solutions for each problem. By default, k is set to 1. 

#### HumanEval 
[HumanEval](https://github.com/openai/human-eval) is a dataset of 164 coding problems from the HumanEval benchmark. Each problem contains a function signature, a canonical solution, and a set of unit tests. 

You can load the dataset using the following code: 
```python
from evoagentx.benchmark import HumanEval
humaneval_dataset = HumanEval() # optional: path="/path/to/save_data"
test_data = humaneval_dataset.get_test_data()
```
Each example in the dataset is in the following format: 
```json
{
    "task_id": "HumanEval/0", 
    "prompt": "the prompt of the problem", 
    "entry_point": "the name of the function to be tested",
    "canonical_solution": "the canonical solution of the problem", 
    "test": "the unit tests of the problem"
}
```

#### MBPP 
[Mostly Basic Python Problems (MBPP)](https://github.com/google-research/google-research/tree/master/mbpp) consists of hundreds of entry-level Python programming problems. Each problem consists of a task description, code solution and 3 automated test cases. We use the [sanitized subset](https://github.com/google-research/google-research/blob/master/mbpp/sanitized-mbpp.json) of the MBPP dataset, which consists of 427 problems with data that are hand-verfied by the authors. To facilitate the evaluation, we convert the MBPP dataset into the HumanEval format. 

You can load the dataset using the following code: 
```python
from evoagentx.benchmark import MBPP
mbpp_dataset = MBPP() # optional: path="/path/to/save_data"
test_data = mbpp_dataset.get_test_data()
```
Each example in the dataset is in the following format, where we keep the original MBPP `task_id`.  
```json
{
    "task_id": 2, 
    "prompt": "the prompt of the problem", 
    "entry_point": "the name of the function to be tested",
    "canonical_solution": "the canonical solution of the problem", 
    "test": "the unit tests of the problem"
}
```
You can also access the original MBPP attributes such as "code", "test_list" in the example by using `example["code"]`. 


#### LiveCodeBench
[LiveCodeBench](https://livecodebench.github.io/) is a contamination-free evaluation benchmark of LLMs for code that continuously collects new problems over time. Particularly, LiveCodeBench also focuses on broader code-related capabilities, such as code execution, and test output prediction, beyond mere code generation. Currently, LiveCodeBench hosts over three hundred high-quality coding problems published between May 2023 and February 2024. 

You can load the dataset using the following code, where `scenario` can be one of [`code_generation`, `test_output_prediction`, `code_execution`] indicating different tasks. `version` denotes different versions of the code generation datasets, which is only available for `code_generation` scenario, and can be one of `["release_v1", "release_v2", "release_v3", "release_v4", "release_v5", "release_latest"]`. Please refer to the [LiveCodeBench](https://livecodebench.github.io/) repository for more details. 

```python
from evoagentx.benchmark import LiveCodeBench
livecodebench_dataset = LiveCodeBench(scenario="code_generation", version="release_v1") # optional: path="/path/to/save_data"
test_data = livecodebench_dataset.get_test_data()
```
