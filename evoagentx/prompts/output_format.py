JSON_SCHEMA_OUTPUT_FORMAT = """
### Outputs Format
You must strictly follow the JSON schema provided below when generating your output:
```json
{json_schema}
```

Your final response must:
- Include all required fields from the JSON schema.
- Adhere to the correct data types for each field.
- Strictly follow the constraints specified for each field in the JSON schema.
- Format each field as {parse_mode}.

Example:
{example}

Note: For optional fields, you can omit them if they are not necessary.
"""


JSON_SCHEMA_OUTPUT_FORMAT_JSON = """
### Outputs Format
Your final response must be a valid JSON object that strictly follows the following JSON schema:
```json
{json_schema}
```

Example:
{example}
"""
