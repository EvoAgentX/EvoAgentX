
SEM_WORKFLOW = {
    "class_name": "LinearWorkflow",
    "goal": "A general workflow for coding tasks.",
    "tasks": [
        {
            "name": "task_parsing",
            "description": "Parse the user's input goal into a detailed task description.",
            "inputs": [
                {"name": "task_description", "type": "string", "required": True, "description": "The description of the programming task."}
            ],
            "outputs": [
                {"name": "task_summary", "type": "string", "required": True, "description": "A detailed summary of the task."}
            ],
            "prompt": "You are a task parsing agent. Summarize the given programming task for the subsequent code generation. You will NOT return anything except for the task summary.\n\n{task_description}",
            "parse_mode": "str" 
        }, 
        {
            "name": "task_refinement",
            "description": "Refine the task description based on the feedback.",
            "inputs": [
                {"name": "task_description", "type": "string", "required": True, "description": "The description of the task."}, 
                {"name": "task_summary", "type": "string", "required": True, "description": "The summary of the task."}
            ],
            "outputs": [
                {"name": "refined_task_description", "type": "string", "required": True, "description": "The refined task description."}
            ],
            "prompt": "Do you think this summary comprehensively covers all aspects of the given programming task? If not, please refine the summary.\n\nTASK:\n{task_description}\nSUMMARY:\n{task_summary}\nYou will NOT return anything except for the refined task description.",
            "parse_mode": "str"
        }, 
        {
            "name": "code_generation",
            "description": "Generate the code for the given task.",
            "inputs": [
                {"name": "refined_task_description", "type": "string", "required": True, "description": "The refined task description."}
            ], 
            "outputs": [
                {"name": "code", "type": "string", "required": True, "description": "The generated code."}
            ],
            "prompt": "You are a proficient Python programmer tasked with coding solutions based on given problem specifications. Your task is to write Python code according to the following problem description. Ensure that the code reads input from standard input (stdin) and writes output to standard output (stdout).\n\nProblem Description:\n{refined_task_description}. \nYou will NOT return anything except for the Python code.", 
            "parse_mode": "str"
        },
        {
            "name": "code_review",
            "description": "Review the generated code.",
            "inputs": [
                {"name": "refined_task_description", "type": "string", "required": True, "description": "The refined task description."},
                {"name": "code", "type": "string", "required": True, "description": "The generated code."}
            ],
            "outputs": [
                {"name": "code_review", "type": "string", "required": True, "description": "The code review."}
            ],
            "prompt": "You are a critical python code reviewer. Your colleague's code cannot pass the sample test. You will be given the Problem Description followed by the corresponding Generated Code by your colleague. Please give some specific explanations and comments to help your colleague to write code that can pass the sample test.\n\nProblem Description:\n{refined_task_description}\nGenerated Code:\n{code}",
            "parse_mode": "str"
        }, 
        {
            "name": "code_refinement",
            "description": "Refine the generated code based on the feedback.",
            "inputs": [
                {"name": "refined_task_description", "type": "string", "required": True, "description": "The refined task description."},
                {"name": "code", "type": "string", "required": True, "description": "The generated code."},
                {"name": "code_review", "type": "string", "required": True, "description": "The code review."}
            ],
            "outputs": [
                {"name": "refined_code", "type": "string", "required": True, "description": "The refined code."}
            ],
            "prompt": "You are a proficient Python programmer tasked with coding solutions based on given problem specifications. You just generated some codes that cannot pass the sample test. Your role is to regenerate python code that strictly adheres to the specifications, ensuring it reads input from standard input (stdin) and writes output to standard output (stdout). You will be given the Problem Description, the Generated Code, and the Comments and Reasons why your previous code fails. You will NOT return anything except for the program.\n\nProblem Description:\n{refined_task_description}\nGenerated Code:\n{code}\nComments and Reasons:\n{code_review}",
            "parse_mode": "str"
        }
    ]
}