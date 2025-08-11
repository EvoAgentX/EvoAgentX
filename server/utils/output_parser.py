"""
Output parser utilities for EvoAgentX server.

This module provides functions to parse and extract structured data from workflow execution outputs.
It supports multiple parsing strategies including JSON, XML, and custom formats.
"""

import json
import re
from typing import Dict, Any, Optional, List, Union
import logging

# Set up logging
logger = logging.getLogger(__name__)


def parse_workflow_output(output: str) -> Dict[str, Any]:
    """
    Parse workflow execution output using enhanced parsing strategies.
    
    This function attempts to extract structured data from workflow outputs
    using multiple parsing strategies in order of preference.
    
    Args:
        output: The raw output string from workflow execution
        
    Returns:
        Dict containing parsed result with keys:
        - "original_message": Original output message
        - "parsed_json": Extracted structured data (dict or None)
        - "parsing_method": Method used for parsing (str)
        - "confidence": Confidence level of parsing (float, 0.0-1.0)
    """
    if not isinstance(output, str):
        return {
            "original_message": str(output),
            "parsed_json": None,
            "parsing_method": "none",
            "confidence": 0.0
        }
    
    result = {
        "original_message": output,
        "parsed_json": None,
        "parsing_method": "none",
        "confidence": 0.0
    }
    
    # Strategy 1: Try EvoAgentX LLMOutputParser (highest confidence)
    try:
        from evoagentx.models.base_model import LLMOutputParser
        parser = LLMOutputParser.parse(output, parse_mode="json")
        structured_data = parser.get_structured_data()
        if structured_data and len(structured_data) > 0:
            result["parsed_json"] = structured_data
            result["parsing_method"] = "evoagentx_llm_parser"
            result["confidence"] = 0.95
            return result
    except Exception as e:
        logger.debug(f"EvoAgentX LLMOutputParser failed: {e}")
    
    # Strategy 2: Try direct JSON parsing with regex extraction
    try:
        json_data = _extract_json_with_regex(output)
        if json_data:
            result["parsed_json"] = json_data
            result["parsing_method"] = "regex_json_extraction"
            result["confidence"] = 0.85
            return result
    except Exception as e:
        logger.debug(f"Regex JSON extraction failed: {e}")
    
    # Strategy 3: Try to extract JSON from code blocks
    try:
        json_data = _extract_json_from_code_blocks(output)
        if json_data:
            result["parsed_json"] = json_data
            result["parsing_method"] = "code_block_extraction"
            result["confidence"] = 0.80
            return result
    except Exception as e:
        logger.debug(f"Code block extraction failed: {e}")
    
    # Strategy 4: Try XML parsing
    try:
        xml_data = _extract_xml_data(output)
        if xml_data:
            result["parsed_json"] = xml_data
            result["parsing_method"] = "xml_parsing"
            result["confidence"] = 0.75
            return result
    except Exception as e:
        logger.debug(f"XML parsing failed: {e}")
    
    # Strategy 5: Try to extract structured data from markdown
    try:
        markdown_data = _extract_markdown_structure(output)
        if markdown_data:
            result["parsed_json"] = markdown_data
            result["parsing_method"] = "markdown_parsing"
            result["confidence"] = 0.70
            return result
    except Exception as e:
        logger.debug(f"Markdown parsing failed: {e}")
    
    # Strategy 6: Try to extract key-value pairs
    try:
        kv_data = _extract_key_value_pairs(output)
        if kv_data:
            result["parsed_json"] = kv_data
            result["parsing_method"] = "key_value_extraction"
            result["confidence"] = 0.60
            return result
    except Exception as e:
        logger.debug(f"Key-value extraction failed: {e}")
    
    # Strategy 7: If all else fails, create a simple structure
    try:
        result["parsed_json"] = {"workflow_output": output, "raw_content": output}
        result["parsing_method"] = "fallback_simple"
        result["confidence"] = 0.10
    except Exception as e:
        logger.warning(f"Fallback parsing failed: {e}")
    
    return result


def _extract_json_with_regex(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON data from text using regex patterns.
    
    Args:
        text: Input text that may contain JSON
        
    Returns:
        Parsed JSON data or None if extraction fails
    """
    # Look for JSON patterns in the output
    # Pattern for complete JSON objects
    json_pattern = r'\{[^{}]*\}'
    matches = re.findall(json_pattern, text)
    
    if matches:
        # Try to parse the first JSON match
        for match in matches:
            try:
                # Clean up the match
                cleaned_match = _clean_json_string(match)
                parsed = json.loads(cleaned_match)
                return parsed
            except json.JSONDecodeError:
                continue
    
    # Try to find JSON arrays
    array_pattern = r'\[[^\[\]]*\]'
    array_matches = re.findall(array_pattern, text)
    
    if array_matches:
        for match in array_matches:
            try:
                cleaned_match = _clean_json_string(match)
                parsed = json.loads(cleaned_match)
                return {"array_data": parsed}
            except json.JSONDecodeError:
                continue
    
    return None


def _extract_json_from_code_blocks(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from code blocks (```json ... ```).
    
    Args:
        text: Input text that may contain code blocks
        
    Returns:
        Parsed JSON data or None if extraction fails
    """
    # Pattern for code blocks with optional language specification
    code_block_pattern = r'```(?:json)?\s*\n(.*?)\n\s*```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    
    if matches:
        for match in matches:
            try:
                cleaned_match = _clean_json_string(match.strip())
                parsed = json.loads(cleaned_match)
                return parsed
            except json.JSONDecodeError:
                continue
    
    return None


def _extract_xml_data(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract structured data from XML-like content.
    
    Args:
        text: Input text that may contain XML
        
    Returns:
        Extracted data as dictionary or None if extraction fails
    """
    try:
        import xml.etree.ElementTree as ET
        from xml.etree.ElementTree import ParseError
        
        # Find XML content
        xml_pattern = r'<(\w+)>(.*?)</\1>'
        matches = re.findall(xml_pattern, text, re.DOTALL)
        
        if matches:
            result = {}
            for tag, content in matches:
                result[tag] = content.strip()
            return result
            
    except ImportError:
        # XML parsing not available, try simple tag extraction
        xml_pattern = r'<(\w+)>(.*?)</\1>'
        matches = re.findall(xml_pattern, text, re.DOTALL)
        
        if matches:
            result = {}
            for tag, content in matches:
                result[tag] = content.strip()
            return result
    
    return None


def _extract_markdown_structure(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract structured data from markdown content.
    
    Args:
        text: Input text that may contain markdown
        
    Returns:
        Extracted data as dictionary or None if extraction fails
    """
    # Extract headers
    header_pattern = r'^(#{1,6})\s+(.+)$'
    headers = re.findall(header_pattern, text, re.MULTILINE)
    
    # Extract lists
    list_pattern = r'^[\s]*[-*+]\s+(.+)$'
    list_items = re.findall(list_pattern, text, re.MULTILINE)
    
    # Extract code blocks
    code_pattern = r'```(\w+)?\n(.*?)```'
    code_blocks = re.findall(code_pattern, text, re.DOTALL)
    
    if headers or list_items or code_blocks:
        result = {}
        
        if headers:
            result["headers"] = [{"level": len(level), "text": text.strip()} for level, text in headers]
        
        if list_items:
            result["list_items"] = [item.strip() for item in list_items]
        
        if code_blocks:
            result["code_blocks"] = [{"language": lang or "text", "code": code.strip()} for lang, code in code_blocks]
        
        return result
    
    return None


def _extract_key_value_pairs(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract key-value pairs from text.
    
    Args:
        text: Input text that may contain key-value pairs
        
    Returns:
        Extracted key-value pairs as dictionary or None if extraction fails
    """
    # Pattern for key: value or key=value
    kv_patterns = [
        r'^[\s]*([^:\n]+):\s*(.+)$',  # key: value
        r'^[\s]*([^=\n]+)=\s*(.+)$',  # key=value
        r'^[\s]*([^:\n]+):\s*$',      # key: (empty value)
    ]
    
    result = {}
    lines = text.split('\n')
    
    for line in lines:
        for pattern in kv_patterns:
            match = re.match(pattern, line)
            if match:
                key = match.group(1).strip()
                value = match.group(2).strip() if len(match.groups()) > 1 else ""
                
                # Clean up the key and value
                key = re.sub(r'[^\w\s-]', '', key).strip()
                if key and key not in result:
                    result[key] = value
    
    return result if result else None


def _clean_json_string(json_str: str) -> str:
    """
    Clean up a JSON string for parsing.
    
    Args:
        json_str: Raw JSON string that may need cleaning
        
    Returns:
        Cleaned JSON string
    """
    # Remove common problematic characters
    cleaned = json_str.strip()
    
    # Remove trailing commas
    cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
    
    # Fix common quote issues
    cleaned = re.sub(r'([{,])\s*([a-zA-Z_]\w*)\s*:', r'\1"\2":', cleaned)
    
    # Remove comments (if any)
    cleaned = re.sub(r'//.*$', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
    
    return cleaned


def parse_structured_output(output: str, output_format: str = "auto") -> Dict[str, Any]:
    """
    Parse output with a specific format preference.
    
    Args:
        output: The output string to parse
        output_format: Preferred output format ("auto", "json", "xml", "markdown", "key_value")
        
    Returns:
        Parsed output data
    """
    if output_format == "auto":
        return parse_workflow_output(output)
    
    result = {
        "original_message": output,
        "parsed_json": None,
        "parsing_method": f"forced_{output_format}",
        "confidence": 0.0
    }
    
    try:
        if output_format == "json":
            json_data = _extract_json_with_regex(output)
            if json_data:
                result["parsed_json"] = json_data
                result["confidence"] = 0.85
        elif output_format == "xml":
            xml_data = _extract_xml_data(output)
            if xml_data:
                result["parsed_json"] = xml_data
                result["confidence"] = 0.75
        elif output_format == "markdown":
            md_data = _extract_markdown_structure(output)
            if md_data:
                result["parsed_json"] = md_data
                result["confidence"] = 0.70
        elif output_format == "key_value":
            kv_data = _extract_key_value_pairs(output)
            if kv_data:
                result["parsed_json"] = kv_data
                result["confidence"] = 0.60
    except Exception as e:
        logger.warning(f"Failed to parse with format {output_format}: {e}")
    
    return result


def validate_parsed_output(parsed_data: Dict[str, Any], expected_schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Validate parsed output against an expected schema.
    
    Args:
        parsed_data: The parsed output data
        expected_schema: Optional schema to validate against
        
    Returns:
        Validation result with validation status and any errors
    """
    validation_result = {
        "is_valid": True,
        "errors": [],
        "warnings": []
    }
    
    if not parsed_data.get("parsed_json"):
        validation_result["is_valid"] = False
        validation_result["errors"].append("No structured data found in output")
        return validation_result
    
    if expected_schema:
        # Basic schema validation
        parsed_json = parsed_data["parsed_json"]
        
        for key, expected_type in expected_schema.items():
            if key not in parsed_json:
                validation_result["warnings"].append(f"Expected key '{key}' not found")
            elif not isinstance(parsed_json[key], expected_type):
                validation_result["errors"].append(
                    f"Key '{key}' has type {type(parsed_json[key]).__name__}, expected {expected_type.__name__}"
                )
                validation_result["is_valid"] = False
    
    return validation_result



