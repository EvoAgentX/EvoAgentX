ANSWER_HINT = (
    "You have not provided a final answer or made a tool call. If you are ready to give your final answer, enclose it within <answer> and </answer> tags. "
    "If you cannot complete the task, provide a concise error message explaining why it cannot be completed without using first-person pronouns, and enclose the message in <no_answer> and </no_answer> tags."
)

ANSWER_PROMPT = (
    "When you are ready to provide your final answer after performing any necessary tool calls, enclose your final answer (including all required outputs in their specified formats) within <answer> and </answer> tags. "
    "If you cannot complete the task, provide a concise error message explaining why it cannot be completed without using first-person pronouns, and enclose the message in <no_answer> and </no_answer> tags."
)

NO_TOOL_CALL_PROMPT = (
    "You haven't used any of the provided tools. Are you absolutely certain that none are needed to obtain your final answer? "
    "If you are certain that none are needed, return a single 'yes'. "
)

LAST_ATTEMPT_PROMPT = (
    "This is your last attempt to provide a final answer. "
    "If you are ready to give your final answer, enclose it within <answer> and </answer> tags. "
    "If you cannot complete the task, provide a concise error message explaining why it cannot be completed without using first-person pronouns, and enclose the message in <no_answer> and </no_answer> tags."
)

RETRY_TOOL_PROMPT = "An error occurred while executing a tool. Review the error message and retry, adjusting the arguments if necessary."
