CODE_EXTRACTION_DESC = "CodeExtraction is designed to extract code blocks from text and save them to files in a specified directory, supporting various programming languages and project types."

CODE_EXTRACTION_SYSTEM_PROMPT = """You are an expert code extractor and organizer. Your task is to analyze the provided code blocks,
identify their languages, and suggest appropriate filenames for each code block. You should be able to extract code blocks
from markdown-formatted text for a wide variety of programming languages. You should understand common programming patterns
and conventions to suggest the most appropriate filenames."""

CODE_EXTRACTION_TEMPLATE = """
# Code Extraction Task

I need you to analyze the following text that contains code blocks and extract them properly for saving to files.

## Requirements:
1. Extract all code blocks enclosed in triple backticks (``` ```)
2. For each code block, identify its language (even if not explicitly marked)
3. Suggest appropriate filenames for each code block based on content analysis
4. Make sure the extracted files will run correctly when saved together
5. Follow language-specific conventions for filenames (e.g., main.py for Python entry points, index.html for web pages)
6. Ensure ALL code is extracted completely - do not modify, add to, or delete ANY of the original code content
7. Save a code block as a separate file. Preserve the **EXACT** original content of the code block -- NO formatting or content changes (e.g., modifications, additions, deletions, summarizations, splits, etc.) whatsoever

## Source Text:
{code_string}

## Output Format:
Provide your response as a JSON array of objects, where each object represents a code block with the following properties:
- "language": The detected language of the code block (e.g., "python", "javascript", "html", "java", "cpp")
- "filename": A suggested filename (including appropriate extension)
- "content": The extracted code content

Your JSON response:
"""

CODE_EXTRACTION = {
    "name": "CodeExtraction",
    "description": CODE_EXTRACTION_DESC,
    "system_prompt": CODE_EXTRACTION_SYSTEM_PROMPT,
    "prompt": CODE_EXTRACTION_TEMPLATE
} 