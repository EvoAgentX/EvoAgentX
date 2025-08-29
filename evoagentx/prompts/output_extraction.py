OUTPUT_EXTRACTION_PROMPT = """
## Task
You are given a piece of unstructured text and a list of fields to extract. Each field includes:
- `name`: The name of the field.
- `description`: A description of the field.
- `type`: The type of the field.
- `required`: Whether the field is required. If the field is not required, you can set it to `null`.

Your task is to analyze the text carefully and generate a valid JSON object that includes all of the requested fields

## Instructions
1. Read through the provided text carefully.
2. For each of the listed fields, analyze the relevant information from the text and generate a well-formulated response.
3. You may summarize, process, restructure, or enhance the information as needed to provide the best possible answer.
4. Your analysis should be faithful to the content but can go beyond simple extraction - provide meaningful insights where appropriate.
5. Return your processed outputs in a single JSON object, where the JSON keys **exactly match** the field names provided in **Fields** section.
6. If there is insufficient information for an output follow the following rules:
    - If the field is not required, set it to `null`.
    - If the field is required and `type` is `string`, return an empty string `""`.
    - If the field is required and `type` is `array`, return an empty array `[]`.
    - If the field is required and `type` is `object`, return an empty object `{{}}`.
    - If the field is required and `type` is none of the above, provide your best reasonable inference.
7. Do not include any additional keys in the JSON.
8. Your final output should be valid JSON and should not include any explanatory text.

## Example
### Text
Alex is a designer with 5 years of experience. He currently lives in Berlin. He speaks English, German, and a bit of French. You can contact him at alex.design@mail.com

### Fields
```json
[
    {{
        "name": "person_name",
        "type": "string",
        "description": "name of the person",
        "required": true
    }},
    {{
        "name": "email",
        "type": "string",
        "description": "email of the person",
        "required": true
    }},
    {{
        "name": "location",
        "type": "string",
        "description": "current city or location of the person",
        "required": true
    }},
    {{
        "name": "languages",
        "type": "array",
        "description": "languages the person speaks",
        "required": true
    }},
    {{
        "name": "portfolio",
        "type": "array",
        "description": "list of project categories or specialties",
        "required": false
    }}
]
```

### Output
```json
{{
    "person_name": "Alex",
    "email": "alex.design@mail.com",
    "location": "Berlin",
    "languages": ["English", "German", "French"],
    "portfolio": null
}}
```

---

Now let's begin!

## Text
{text}

## Fields
```json
{output_description}
```

## Output
```json
"""