
def extract_test_output_code(model_output: str):
    """Extract test code from model output, prioritizing assert statements.
    
    This function tries to extract test output code from the model's text output
    using two strategies:
    1. Find the last line that starts with "assert"
    2. If no assert lines are found, look for code blocks within ```python or ``` markers
    
    Args:
        model_output: Raw string output from the model
        
    Returns:
        String containing either an assert statement or code block contents,
        or an empty string if nothing is found
    """
    outputlines = model_output.split("\n")
    # find the last line startwith assert...
    indexlines = [i for i, line in enumerate(outputlines) if line.startswith("assert")]
    if indexlines:
        return outputlines[indexlines[-1]]
    
    # first try to extract ```python if not then try ```
    indexlines = [
        i
        for i, line in enumerate(outputlines)
        if "```python" in line or "```Python" in line
    ]
    if indexlines:
        start_index = indexlines[0]
    else:
        start_index = None
    indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
    if start_index is not None:
        indexlines = [i for i in indexlines if i > start_index]
        indexlines = [start_index] + indexlines

    if len(indexlines) < 2:
        return ""
    return "\n".join(outputlines[indexlines[0] + 1 : indexlines[1]])


def extract_execution_code(model_output: str, cot: bool = False):
    """Extract execution code from model output for the code execution task.
    
    This function extracts the actual code execution output from the model's response,
    handling various formatting patterns including chain-of-thought (CoT) responses
    and delimiters like "==" or "[ANSWER]"/"[/ANSWER]" tags.
    
    Args:
        model_output: Raw string output from the model
        cot: Boolean indicating whether the output uses chain-of-thought format
            with "[ANSWER]" tag (default: False)
        
    Returns:
        String containing the cleaned execution code
    """
    if cot:
        if "[ANSWER]" in model_output:
            model_output = model_output.split("[ANSWER]")[1].strip()
    if "==" in model_output:
        model_output = model_output.split("==")[1].strip()
    if "[/ANSWER]" in model_output:
        model_output = model_output.split("[/ANSWER]")[0].strip()
    else:
        model_output = model_output.split("\n")[0].strip()
    return model_output.strip()