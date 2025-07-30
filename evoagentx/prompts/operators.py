# Acknowledgement: Modified from AFlow (https://github.com/geekan/MetaGPT/blob/main/metagpt/ext/aflow/scripts/prompts/prompt.py) under MIT License

ANSWER_GENERATION_PROMPT = """
Think step by step and solve the problem.
1. In the "thought" field, explain your thinking process in detail.
2. In the "answer" field, provide the final answer concisely and clearly. The answer should be a direct response to the question, without including explanations or reasoning.
You should format your output in xml format. For example, ouptut the thoughts in <thought>xxx</thought> format. 

Your task: {input}
"""

QA_SC_ENSEMBLE_PROMPT = """
Several answers have been generated to a same question. They are as follows:
{solutions}

Identify the concise answer that appears most frequently across them. This consistency in answers is crucial for determining the most reliable solution.

In the "thought" field, provide a detailed explanation of your thought process. In the "solution_letter" field, output only the single letter ID (A, B, C, etc.) corresponding to the most consistent solution. Do not include any additional text or explanation in the "solution_letter" field.

You should format your output in xml format. For example, ouptut the thoughts in <thought>xxx</thought> format. 
"""

REFLECTION_ON_PUBLIC_TEST_PROMPT = """
Given a code problem and a python code solution which failed to pass test or execute, you need to analyze the reason for the failure and propose a better code solution.: 
### problem
{problem}

### Code Solution
{solution}

### Execution Result
{exec_pass}

#### Failed Test Case
{test_fail}

Please provide a reflection on the failed test cases and code solution, followed by a better code solution without any additional text or test cases.

Your response should be formatted as follows:
```json
{{
  "reflection_and_solution": "Your improved code solution here"
}}
```

Only include the improved code in the reflection_and_solution field, without any additional explanations or comments outside the code.
"""

SC_ENSEMBLE_PROMPT = """
Given the question described as follows: {problem}
Several solutions have been generated to address the given question. They are as follows:
{solutions}

Carefully evaluate these solutions and identify the answer that appears most frequently across them. This consistency in answers is crucial for determining the most reliable solution.

In the "thought" field, provide a detailed explanation of your thought process. In the "solution_letter" field, output only the single letter ID (A, B, C, etc.) corresponding to the most consistent solution. Do not include any additional text or explanation in the "solution_letter" field.

You should format your output in xml format. For example, ouptut the thoughts in <thought>xxx</thought> format. 
"""

PYTHON_CODE_VERIFIER_PROMPT = """
You are a professional Python programmer. Your task is to write complete, self-contained code based on a given mathematical problem and output the answer. The code should include all necessary imports and dependencies, and be ready to run without additional setup or environment configuration.

Problem description: {problem}
Other analysis: {analysis}
{feedback}

Your code should:
1. Implement the calculation steps described in the problem.
2. Define a function named `solve` that performs the calculation and returns the result. The `solve` function should not require any input parameters; instead, it should obtain all necessary inputs from within the function or from globally defined variables.
3. `solve` function return the final calculation result.

Please ensure your code is efficient, well-commented, and follows Python best practices. The output should be limited to basic data types such as strings, integers, and floats. It is prohibited to transmit images or other file formats. The code output is intended for a text-based language model.
"""


PREDICTOR_PROMPT = """
Let's think step by step.

Question: {problem}
{context}
Reasoning: Let's think step by step in order to produce the answer. We ...
Answer: 

You should format your output in xml format. For example, ouptut the answer in <reasoning>xxx</reasoning> and <answer>xxx</answer> format. 

"""

REFLECTOR_PROMPT = """
Please review the answer and crticize on where might be wron. If you are absolutely sure it is correct, output 'True' in 'correctness'.

Question: {problem}
{context}
Text: {text}
Reasoning: Let's think step by step in order to produce the correctness. We ...
Feedback:
Correctness: True/False indicating if answer is correct given the question

You should format your output in xml format. For example, ouptut the answer in <reasoning>xxx</reasoning>, <feedback>xxx</feedback> and <correctness>xxx</correctness> format. 
"""

REFINER_PROMPT = """
Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better. Show your final answer bracketed between <answer> and </answer> at the end.

---
Question: {problem}
{context}
Previous answer: {previous_answer}
Reflection: {reflection}
Correctness: {correctness}
Reasoning: 
Answer: 

Please output your reasoning and final answer in the following xml format:
<reasoning>xxx</reasoning>
<answer>xxx</answer>
"""

SUMMARIZER_PROMPT = """"
Based on the question, retrive relevant information from context that is ONLY helpful in answering the question. Include all key information. Do not repeat context

Question: {problem}
Context: {context}
Summary: Only generate the summary. Start with Summary:

You should format your output in xml format. For example, ouptut the answer in <summary>xxx</summary>
"""

DEBATER_PROMPT = """
These are the solutions to the question from other agents. Examine the solutions from other agents in your rationale, finish by giving an updated answer. Let's think step by step. Provide a complete and correct code implementation in python.

---
Question: {problem}
{context}
Solutions: {solutions}
Reasoning: Let's think step by step in order to examine the solutions from other agents. We ...

index: You should return the index of the solution you choose to refine.
Answer: 

Please output your reasoning and final answer in the following xml format:
<index>xxx</index>
<reasoning>xxx</reasoning>
<answer>xxx</answer>
"""

CODE_REFLECTOR_PROMPT = """
Please determine the correctness of the solution in passing all test cases . If it
fails , based on the error message and trackback , think step by step , carefully
propose an updated solution in the answer output with a correct code
implementation in python .

---
Question: {question}
Previous Solution: {previous_solution}
## Traceback
It contains the test cases, execution results, and ground truth. If there is an error, the relevant traceback is given.
Traceback: {traceback}
## Correctness: 'True False' based on the correctness of executive feedback. If there
is an error message, output 'False' 
Correctness:
Reasoning:

You should format your output in xml format. For example, ouptut the answer in <reasoning>xxx</reasoning>, <correctness>xxx</correctness> and <answer>xxx</answer> format. 
"""
