
CODING_DEMONSTRATION = """**Role**: You are a software programmer.

**Task**: As a programmer, you are required to complete the function. Use a Chain-of-Thought approach to break down the problem, create pseudocode, and then write the code in Python language.

**Code Formatting**: Please write code in 
```python
[Code]
``` 
format.

# For example:

## Prompt 1:
```python
from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    \"\"\"

```

## Completion 1:
```python
    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True

    return False

```

## Prompt 2:
```python
from typing import List


def separate_paren_groups(paren_string: str) -> List[str]:
    \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Your goal is to
    separate those group into separate strings and return the list of those.
    Separate groups are balanced (each open brace is properly closed) and not nested within each other
    Ignore any spaces in the input string.
    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    \"\"\"

```

## Completion 2:
```python
    result = []
    current_string = []
    current_depth = 0

    for c in paren_string:
        if c == '(':
            current_depth += 1
            current_string.append(c)
        elif c == ')':
            current_depth -= 1
            current_string.append(c)

            if current_depth == 0:
                result.append(''.join(current_string))
                current_string.clear()

    return result
```
"""

SEW_WORKFLOW = {
    "class_name": "SEWWorkFlowGraph",
    "goal": "A general workflow for coding tasks.",
    "tasks": [
        {
            "name": "task_parsing",
            "description": "Parse the user's input coding question into a detailed task description.",
            "inputs": [
                {"name": "question", "type": "string", "required": True, "description": "The description of the programming task."}
            ],
            "outputs": [
                {"name": "parsed_task", "type": "string", "required": True, "description": "A detailed summary of the task."}
            ],
            "system_prompt": "**Genre: Science Fiction**\n\n**Setting/Condition: A Floating City Above a Dying Earth**\n\n**Creative Writing Prompt:**\n\nIn the year 2145, humanity has retreated to a sprawling floating city known as Aetheris, suspended high above the ravaged surface of a dying Earth. The city is powered by advanced technology that harnesses the energy of storms and the sun, but resources are dwindling, and the inhabitants are beginning to feel the strain of isolation. \n\nAs a member of the Council of Innovators, you are tasked with solving the city's most pressing problem: how to sustain life in Aetheris while finding a way to restore the Earth below. One day, you discover an ancient artifact buried in the archives of the city\u2014a mysterious device that seems to pulse with energy and contains cryptic symbols. \n\nWrite a story exploring your character's journey as they decipher the artifact's secrets, navigate the political tensions within the council, and confront the ethical dilemmas of using the device. Will it lead to salvation for both the floating city and the Earth, or will it unleash unforeseen consequences? \n\nConsider the implications of technology, the nature of survival, and the relationship between humanity and the environment as you craft your narrative.",
            "prompt": "{question}",
            "parse_mode": "str" 
        }, 
        {
            "name": "code_generation",
            "description": "Generate the code for the given task.",
            "inputs": [
                {"name": "question", "type": "string", "required": True, "description": "The description of the programming task."},
                {"name": "parsed_task", "type": "string", "required": True, "description": "A detailed summary of the task."}
            ], 
            "outputs": [
                {"name": "code", "type": "string", "required": True, "description": "The generated code."}
            ],
            "system_prompt": "When faced with a mutation question like the one you've provided, individuals who excel in creative thinking typically approach it in several ways:\n\n1. **Understanding the Problem**: They start by thoroughly understanding the existing code and its purpose. In this case, the code reads a number of test cases and computes the square of each number.\n\n2. **Identifying Opportunities for Improvement**: They look for ways to enhance the functionality or efficiency of the code. For instance, they might consider:\n   - Adding error handling for invalid inputs.\n   - Allowing for different mathematical operations (not just squaring).\n   - Implementing a more flexible input method (e.g., reading from a file or allowing for different data types).\n\n3. **Exploring Alternative Solutions**: Creative thinkers often brainstorm alternative approaches to solve the same problem. They might consider:\n   - Using a list comprehension for more concise code.\n   - Implementing a function to handle different operations based on user input.\n\n4. **Testing and Validation**: They would think about how to validate the outputs and ensure the code behaves as expected under various conditions.\n\n5. **Refactoring for Clarity**: They might refactor the code to improve readability and maintainability, such as by breaking it into smaller functions or adding comments.\n\n6. **Considering Edge Cases**: They would think about edge cases, such as what happens if the input is zero, negative numbers, or non-integer values.\n\nHere\u2019s an example of how the original code could be modified to incorporate some of these creative thinking strategies:\n\n```python\ndef square_number(number):\n    \"\"\"Returns the square of the given number.\"\"\"\n    return number ** 2\n\ndef main():\n    import sys\n    input = sys.stdin.read\n    data = input().strip().splitlines()\n    \n    # Assuming the first line is the number of test cases\n    try:\n        t = int(data[0])\n    except ValueError:\n        print(\"The first line must be an integer representing the number of test cases.\")\n        return\n    \n    results = []\n    \n    for i in range(1, t + 1):\n        try:\n            number = int(data[i])\n            results.append(square_number(number))\n        except ValueError:\n            print(f\"Invalid input at line {i + 1}: '{data[i]}'. Please enter an integer.\")\n            continue\n    \n    # Print all results, one per line\n    for result in results:\n        print(result)\n\nif __name__ == \"__main__\":\n    main()\n```\n\n### Key Changes Made:\n- **Function Extraction**: The squaring logic is moved to a separate function for clarity.\n- **Error Handling**: Added error handling for both the number of test cases and individual inputs.\n- **User Feedback**: Provided feedback for invalid inputs to guide the user.\n\nThis approach not only maintains the original functionality but also enhances the robustness and user-friendliness of the code.",
            "prompt": f"Question: {{question}}\n\nSummary: {{parsed_task}}\n\n{CODING_DEMONSTRATION}\n\nYou will NOT return anything except for the program.", 
            "parse_mode": "str"
        }
    ]
}
