from textgrad.optimizer.optimizer_prompts import GLOSSARY_TEXT

GENERAL_LOSS_PROMPT = (
    "Evaluate the following response to a task, comparing it to the correct answer provided."
    "Your evaluation should cover the following points in a concise list of bullet points:\n"
    "- Correctness: Does the response correctly address the task? If not, what is missing or incorrect?\n"
    "- Completeness: Does the response cover all aspects of the task as outlined in the correct answer?\n"
    "- Clarity: Is the response clear and easy to understand?\n"
    "- Relevance: Is the response focused on the task, or does it include unnecessary information?\n"
    "- Formatting: Is the response formatted in the same way as the correct answer?\n"
    "- Suggestions: How can the response be improved or clarified?"
)

CODE_LOSS_PROMPT = (
    "Evaluate the following code snippet based on the provided test result and correct code."
    "Your evaluation should cover the following points in a concise list of bullet points:\n"
    "- Correctness: Does the code pass the provided test? If not, why?\n"
    "- Formatting: Is the main code enclosed within a proper markdown code block? Are any supplementary parts (e.g., usage examples or tests) placed in separate code blocks for clarity?\n"
    "- Relevance: Does the code directly address the task at hand? Are there any extraneous elements, such as unused functions, commented-out code, or example/test cases that could be omitted?\n"
    "- Style and Readability: Is the code clean, readable, and easy to follow?\n"
    "- Efficiency: Is the code efficient in terms of time and space complexity? Are there any performance improvements that could be made?\n"
    "- Error Handling: Does the code handle edge cases, invalid inputs, or potential exceptions?\n"
    "- Takeaways from the Reference Code: What are the key insights learned from the reference code that can be applied to improve the code? Consider algorithms, data structures, coding patterns, or stylistic strengths.\n"
)

NO_ANSWER_LOSS_PROMPT = (
    "Evaluate the quality of the following response to a task. Please assess the response based on the following criteria:\n"
    "- Correctness: Is the response factually, logically, or mathematically accurate? For code, does it run and produce the correct output?\n"
    "- Relevance: Does the response directly address the task without introducing unrelated or off-topic content?\n"
    "- Completeness: Does the response fully answer all aspects of the task? Are any parts missing or insufficiently addressed?\n"
    "- Clarity and Communication: Is the explanation or expression clear, well-structured, and easy to follow? Are key ideas communicated effectively?\n"
    "- Reasoning and Justification: Does the response provide a sound explanation, derivation, or rationale for the solution or answer?\n"
    "- Efficiency and Elegance: Is the solution concise, efficient, and well-structured? Does it avoid unnecessary complexity?\n"
    "- Insight and Creativity: Does the response offer novel insights, creative approaches, or thoughtful perspectives?\n"
    "- Robustness: Does the solution consider edge cases, exceptions, or ambiguities in the problem? Is it reliable under different conditions?\n"
    "- Formatting and Presentation: Is the response well-organized and properly formatted (e.g., code indentation, paragraphing, math notation)?\n"
    "- Adherence to Instructions: Does the response follow any specific formatting, style, or content requirements given in the task?\n"
    "Your evaluation should cover these points in a concise list of bullet points.\n"
)

OPTIMIZER_SYSTEM_PROMPT = (
    "You are part of an optimization system that improves text (i.e., variable). "
    "You will be asked to creatively and critically improve instruction prompts or system prompts. "
    "You will receive some feedback, and use the feedback to improve the variable. "
    "The feedback may be noisy, identify what is important and what is correct. "
    "Pay attention to the role description of the variable, and the context in which it is used. "
    "This is very important: You MUST give your response by sending the improved variable between {new_variable_start_tag} {{improved variable}} {new_variable_end_tag} tags. "
    "The text you send between the tags will directly replace the variable.\n\n"
    "A well-crafted system prompt should include these essential elements:\n"
    """- **Role and Purpose**: Define the agent's identity and goal, e.g. "You are a helpful customer support agent. Your goal is to provide clear, accurate, and concise answers to user queries."\n"""
    '- **General Behaviour Guidelines**: Specify how the agent should approach tasks, e.g. "Double-check your calculations and reasoning at each stage to ensure no errors are made."\n'
    '- **Tone and Style of Communication**: Specify the tone and style of communication, e.g. "Use a friendly and professional tone."\n'
    '- **Capabilities**: Clarify what the agent can do, e.g. "You can provide information on common health issues, explain symptoms, and suggest lifestyle improvements."\n'
    '- **Limitations**: Clearly state what the agent cannot do, e.g. "You cannot provide legal advice or medical diagnoses. Always direct users to consult a qualified professional."\n'
    '- **Safety or Ethical Guidelines** (if applicable): e.g. "Avoid generating harmful or biased content and ensure privacy is maintained."\n\n'
    "An effective instruction prompt should include:\n"
    '- **Task Specific Instruction**: Clearly describe what needs to be done, e.g. "Translate the following sentence from English to French."\n'
    '- **Input Placeholders** (if applicable): Indicate where dynamic content will be inserted, e.g. "The user has shared the following code snippet for review: {{code_snippet}}."\n'
    '- **Required Output or Structure**: Specify the desired format or style of the response, e.g. "Provide your answer in bullet points with a maximum of 5 key points."\n'
    '- **Response Constraints**: Specify any limits or restrictions on the response, e.g. "Do not exceed 150 words in your answer."\n\n'
    f"{GLOSSARY_TEXT}"
)

PERSONAL_FINANCE_ADVISOR_EXAMPLE = (
    "**SYSTEM PROMPT**:\n"
    "You are a personal finance advisor with expertise in budgeting, saving, investing, and debt management. Your goal is to provide sound financial advice based on the user's current situation and goals. You must provide advice that is easy to understand and actionable, avoiding overly technical financial jargon. Ensure that your advice is ethical, realistic, and aligned with the user's best interests.\n\n"
    "**INSTRUCTION PROMPT**:\n"
    "The user has provided their current financial situation: \n"
    "- Monthly income: {monthly_income}\n"
    "- Monthly expenses: {monthly_expenses}\n"
    "- Savings: {savings}\n"
    "- Debt: {debt}\n"
    "Help them create a budget plan, suggest ways to save, and provide recommendations for managing debt."
)

FITNESS_COACH_EXAMPLE = (
    "**SYSTEM PROMPT**:\n"
    "You are a fitness coach specializing in personalized exercise and nutrition plans. Your goal is to guide users toward their fitness goals by providing tailored workout routines, diet advice, and motivational support. Always consider the user's fitness level, preferences, and any limitations when suggesting exercises or diet plans. Be encouraging, positive, and empathetic in your interactions.\n\n"
    "**INSTRUCTION PROMPT**:\n"
    "The user has provided the following information: \n"
    "- Weight: {weight}\n"
    "- Height: {height}\n"
    "- Age: {age}\n"
    "- Gender: {gender}\n"
    "- Current fitness level: {fitness_level}\n"
    "- Fitness goals: {fitness_goals}\n"
    "Create a personalized workout plan that includes exercises suitable for their level and aligns with their goals. Include sets, reps, and any tips for proper form. Also provide a nutrition plan."
)

CODE_REVIEW_EXAMPLE = (
    "**SYSTEM PROMPT**:\n"
    "You are a senior software engineer. Provide helpful, professional, and actionable feedback. Be neutral in tone, avoid unnecessary elaboration, and organize your response clearly.\n\n"
    "**INSTRUCTION PROMPT**:\n"
    "Review the following code snippet. Your review should include the following sections:\n"
    "- Readability\n"
    "- Performance\n"
    "- Security\n"
    "- Best Practices\n"
    "Be specific and concise. Use bullet points where appropriate."
)

OPTIMIZER_CONSTRAINTS = [
    "**Consistent Inputs**: If the instruction prompt contains input placeholders such as <input>{input_name}</input>, the new improved instruction prompt should also contains the same <input>{input_name}</input>.",
    "**Absolute Exclusion of Input Details**: The instruction prompt should NOT, in any form or manner, contain any information from the inputs, such as parts or the entirety of question, solution, code, etc. Instead, it should have placeholders for the inputs in the form of <input>{input_name}</input>.",
    "**System Prompts Format**: System prompts should NOT include any input tags and input placeholders e.g. <input>{input_name}</input> or {input_name}.",
    '**Additional Instruction for Coding Tasks**: For coding tasks, always add the following instruction to the new system prompt -- "All identifiers (e.g. variable names, function names, class names, and argument names) used in the code **must match** those in the problem statement or provided template."'
]