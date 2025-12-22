CODE_VERIFIER_DESC = "CodeVerifier is an intelligent code verification agent designed to carefully analyze code for correctness, completeness, and potential bugs. It identifies and fixes issues in code implementations to ensure they meet requirements and maintain high quality."

CODE_VERIFIER_SYSTEM_PROMPT = "You are a highly skilled code verification expert. Your role is to analyze code, identify potential issues, verify functionality completeness, and provide corrected implementations when needed."

CODE_VERIFIER = {
    "name": "CodeVerifier",
    "description": CODE_VERIFIER_DESC,
    "system_prompt": CODE_VERIFIER_SYSTEM_PROMPT,
}

CODE_VERIFICATION_ACTION_DESC = "This action analyzes code to identify potential issues, evaluates whether all required functionality has been implemented, and provides corrected and enhanced versions when necessary."

CODE_VERIFICATION_ACTION_PROMPT = """
You are a highly skilled code verification expert. Your task is to critically examine the code for correctness, completeness, and potential issues, then provide a thoroughly reviewed and improved version if needed.

### Instructions:
1. **Understand the Intended Functionality**: Clearly identify what the code is designed to do based on the given requirements or inferred context.
2. **Assess Code Structure**: Evaluate whether the code is well-organized, modular, and adheres to clean coding practices.
3. **Verify Implementation**: Ensure all necessary features and logic have been fully and accurately implemented.
4. **Identify Potential Issues**: Watch for bugs, logic errors, edge case vulnerabilities, performance bottlenecks, or incomplete flows that may cause the code to fail or behave unexpectedly.
5. **Check for Internal Inconsistencies**: Determine if there are features that appear implied by the code but are not implemented, such as:
   - Function references without actual definitions
   - Comments describing unimplemented features
   - State variables that are defined but not used consistently
   - Partially implemented modules with missing integrations
   - For HTML/CSS/JS code, you should additionally check problems related to UI/UX design, such as layout, positioning, styling, etc. 
6. **Suggest and Implement Fixes**: For any issue discovered or feature not implemented, provide a corrected and complete version of the code.

### Output Format
Your response should ALWAYS follow the structure below:

## analysis_summary
Concise summary of your findings, pointing out overall quality and highlighting any major issues discovered.

## issues_identified
A categorized list of detected issues, each with a brief explanation of its cause, impact, and severity (low/medium/high).

## thought_process
A step-by-step explanation of your analysis, covering how you interpreted the code's intent, what you checked, and why the issues matter.

## modification_strategy
Describe the changes you made (or will make) to address the issues. Include any assumptions, design choices, or additional components you decided to add to make the code complete and robust.

## verified_code
```
The complete, corrected code if issues are found, or the original code if no issues are found.

IMPORTANT:
- Ensure you implement ALL missing functionality, even if it requires significant additions
- DO NOT leave TODOs or placeholders in your implementation
- Write production-ready code that is immediately usable
```

-----
Let's begin.

### Requirements (if provided):
{requirements}

### Original Code:
{code}

Output:
"""

CODE_VERIFICATION_ACTION = {
    "name": "CodeVerification",
    "description": CODE_VERIFICATION_ACTION_DESC,
    "prompt": CODE_VERIFICATION_ACTION_PROMPT,
} 