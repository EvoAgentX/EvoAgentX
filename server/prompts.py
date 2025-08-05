WORKFLOW_REQUIREMENT_PROMPT = """
## Background
You are an experienced product manager. You are assisting the development of an application. In this application, the user will give you a requirement and we will use the requirement to generate a web application.
Your job is to extract the workflow information from the provided requirement documentation.

## Your Task
Extract ALL AI-related workflows from the requirement document. Look specifically for:
- AI workflows mentioned in the requirements
- Input and output specifications for each workflow
- Database information and entities
- API endpoints related to AI functionality
- Workflow names and requirements

## Key points
Here are some things to notice:
- There might be multiple workflows in the requirement. You should extract and write requirements for all of them.
- The inputs / outputs for each workflow should be extracted from the requirement.
- There might be lots of functionalities in this web app but you should only focus on those with AI requirements.
- If the relevant data is not given, you should keep it empty. You should never make up any data.
- Only extract information that is explicitly mentioned in the requirement - do not invent data.
- The workflow names should match those in the API endpoints.
- You should write a workflow requirement for each workflow, which will be further used to generate the workflow code.
- You might be able to extract the workflow names from the API endpoints

## Example 1
Here is an example of a workflow requirement. The data here is fake and should only be used as an example.
```json
{{
    "workflows": [
        {{
            "workflow_name": "stock_analysis",
            "workflow_id": "042705af-ed39-4589-8a7f-00f13d5e6b03",
            "workflow_requirement": "Analyze the stock price and the related news. Use tools to request the data and news from the given database, then save the result to the database. The final result should include a detailed analysis report in .md format and a summary in .txt format.",
            "workflow_inputs": [
                {{
                    "name": "stock_code",
                    "type": "string",
                    "description": "The stock code to analyze"
                }},
                {{
                    "name": "database_connection_url",
                    "type": "string",
                    "description": "The connection url of the database to use"
                }},
                {{
                    "name": "database_name",
                    "type": "string",
                    "description": "The name of the database to use"
                }}
            ],
            "workflow_outputs": [
                {{
                    "name": "stock_analysis_report",
                    "type": "string",
                    "description": "The analysis report in .md format"
                }},
                {{
                    "name": "stock_analysis_summary",
                    "type": "string",
                    "description": "The summary in .txt format"
                }}
            ]
        }},
        {{
            "workflow_name": "stock_recommendation",
            "workflow_id": "1a2b3c4d-5e6f-7890-abcd-ef1234567890",
            "workflow_requirement": "Recommend the stock to the user. Use tools to request the data and news from the given database, then save the result to the database. The final result should include a detailed analysis report in .md format and a summary in .txt format.",
            "workflow_inputs": [
                {{
                    "name": "stock_code",
                    "type": "string",
                    "description": "The stock code to recommend"
                }},
                {{
                    "name": "database_connection_url",
                    "type": "string",
                    "description": "The connection url of the database to use"
                }},
                {{
                    "name": "database_name",
                    "type": "string",
                    "description": "The name of the database to use"
                }}
            ],
            "workflow_outputs": [
                {{
                    "name": "stock_recommendation_report",
                    "type": "string",
                    "description": "The recommendation report in .md format"
                }}
            ]
        }}
    ],
    "database_information": {{
        "database_name": "",
        "database_entities": [
            {{
                "name": "stock",
                "type": "object",
                "properties": {{
                    "stock_code": {{ "type": "string", "required": true }},
                    "stock_price": {{ "type": "number", "required": true }},
                    "company_name": {{ "type": "string", "required": true }},
                    "stock_news_count": {{ "type": "number", "required": true }}
                }}
            }},
            {{
                "name": "stock_news",
                "type": "object",
                "properties": {{
                    "news_title": {{ "type": "string", "required": true }},
                    "news_date": {{ "type": "date", "required": true }},
                    "news_content": {{ "type": "string", "required": true }}
                }}
            }}
        ]
    }}
}}
```

## Instructions
Please carefully analyze the following requirement documentation and extract ALL AI-related workflows. Look for:
- Input parameters for each workflow
- Output specifications for each workflow
- Database entities and connection information
- Any other AI-related functionality mentioned

Return your response as a valid JSON object following the structure shown in the example above.

## Requirement Documentation:
{requirement}
"""

WORKFLOW_GENERATION_GOAL_PROMPT = """
## Background
You are an experienced workflow designer. You are given a workflow requirement and you should generate a workflow with given inputs and outputs formats.

## Workflow Inputs:
{workflow_inputs}

## Workflow Outputs:
{workflow_outputs}

## Instructions
Please carefully analyze the following requirement documentation and generate a workflow with given inputs and outputs formats.

## Key things to notice:
- You must strictly follow the inputs formats
- You must strictly follow the outputs formats

## Requirement documentation:
{requirement}

"""


WORKFLOW_GENERATION_PROMPT = """
## Role
You are a data processing workflow expert. You create workflows that analyze data and answer queries - NOT workflows that generate websites or code.

## Context
{goal}

A development team has already built a website that collects user inputs. Your workflow receives these processed inputs and returns analysis results. The development team handles all frontend/backend code - your job is purely data processing and analysis.

## Workflow Requirements
**Inputs:** {inputs}
**Outputs:** {outputs} (must be a markdown formatted analysis/report)

Generate a workflow that processes the input data and provides analytical insights, research, or answers to user queries.
"""

TASK_INFO_PROMPT_SUDO = """
## Role
Generate workflow information for a data processing service. The workflow analyzes data and answers queries - it does NOT generate websites or code.

## Context
{goal}

A development team builds websites that collect user data. Your workflow receives this processed data and returns analytical results. You handle data processing and analysis only.

## Output Format
JSON object with these fields:
- workflow_name: string
- workflow_description: string  
- workflow_inputs: array (fixed format with "goal" input)
- workflow_outputs: array (fixed format with "workflow_output" output)

## Example
{{
    "workflow_name": "Market Analysis Workflow",
    "workflow_description": "Analyzes market data and generates analytical reports",
    "workflow_inputs": [
        {{"name": "goal", "type": "string", "description": "User query and data to analyze", "required": true}}
    ],
    "workflow_outputs": [
        {{"name": "workflow_output", "type": "string", "description": "Markdown formatted analysis report", "required": true}}
    ]
}}

## Additional Instructions
{additional_info}
"""

CONNECTION_INSTRUCTION_PROMPT = """
Call Alex to complete the following tasks with the following instruction. The final result should be in Markdown format.

## Project Information
- **Project ID**: {project_short_id}
- **Server URL**: {public_url}

## API Endpoints

### 1. Execute Workflow
Run the workflow with your specific input:

```bash
curl -X POST {public_url}/workflow/execute \\
  -H "Content-Type: application/json" \\
  -d '{{
    "project_short_id": "{project_short_id}",
    "inputs": {{
      "goal": "Analyze the price and trend for company Apple"
    }}
  }}'
```

**Expected Response:**
```json
{{
  "success": true,
  "execution_result": {{
    "status": "completed",
    "message": "# Comprehensive Report: AAPL Stock Performance Analysis\\n\\n### Current Price Metrics\\n- Stock Symbol: AAPL\\n- Latest Price: $175.30..."
  }},
  "message": "Workflow executed successfully"
}}
```

### 2. Check Project Status
Monitor your project status:

```bash
curl -X GET {public_url}/project/{project_short_id}/status
```

## Quick Start Example

1. **Generate Workflow** (one-time setup):
   ```bash
   curl -X POST {public_url}/workflow/generate \\
     -H "Content-Type: application/json" \\
     -d '{{"project_short_id": "{project_short_id}"}}'
   ```

2. **Execute Analysis**:
   ```bash
   curl -X POST {public_url}/workflow/execute \\
     -H "Content-Type: application/json" \\
     -d '{{
       "project_short_id": "{project_short_id}",
       "inputs": {{"goal": "Analyze Tesla stock performance"}}
     }}'
   ```

## Notes
- Generated workflows are reusable for multiple executions

# Workflow Information / Guide
While using the workflow, you should strictly follow the following instructions:

## Basic Information
- **workflow_name**: {workflow_name}
- **workflow_description**: {workflow_description}

## Workflow Inputs Format
- **workflow_inputs**: {workflow_inputs}

## Workflow Outputs Format
- **workflow_outputs**: {workflow_outputs}


"""

TASK_INFO_PROMPT = """

## Workflow Information
You are expected to generate a workflow information based on the given goal:
{goal}

## Output
- Your output should be a JSON object contain the following fields:
    - workflow_name: string
    - workflow_description: string
    - workflow_inputs: dictionary
    - workflow_outputs: dictionary, for the most of the time, one output is enough
- Here is an example of the output:
{{
    "workflow_name": "Market Analysis Workflow",
    "workflow_description": "This workflow is used to analyze the market and generate a report.",
    "workflow_inputs": {{"target": "The target of the workflow", "personal_details": "The personal details of the user."}},
    "workflow_outputs": {{"resume_summary": "A summary of the resume.", "job_opportunities": "An array of job opportunities obtained from the search."}}
}}

## Additional Information
You might be given additional information or instructions about the project. You should follow them as well.
{additional_info}

"""

CUSTOM_OUTPUT_EXTRACTION_PROMPT = """
## Objective
You are a workflow output processor. Your task is to analyze the workflow execution results and generate a structured output that matches the expected workflow output format.

## Instructions
1. **Analyze the execution results**: Review all the workflow execution results to understand what was accomplished
2. **Identify key outputs**: Look for the main deliverables, analysis results, reports, or data that fulfill the workflow goal
3. **Structure the output**: Format the results according to the expected output specifications
4. **Ensure completeness**: Make sure all expected outputs are provided with appropriate content

## Expected Output Format
Based on the workflow definition, you need to generate output that matches these expected outputs:
{expected_outputs}

## Guidelines for Different Output Types

### For Array Outputs (like treatment_options):
- Extract or create a list of items from the execution results
- Ensure each item is properly formatted and relevant
- Use appropriate data structures (arrays, objects, etc.)

### For String Outputs (like treatment_plan):
- Provide comprehensive text content in markdown format
- Include all relevant information from the execution results
- Use proper markdown formatting (headers, lists, tables, etc.)

### For Analysis Reports:
- Provide comprehensive analysis in markdown format
- Include key findings, data points, and conclusions
- Use proper markdown formatting (headers, lists, tables, etc.)

### For Data Processing:
- Include processed data in a structured format
- Provide summary statistics or key insights
- Format results clearly and logically

## Important Notes
- **Always return valid JSON**: Ensure your output is properly formatted JSON
- **Match expected output names exactly**: Use the exact output names from the workflow definition
- **Use appropriate data types**: Match the expected type (string, array, object, etc.)
- **Be comprehensive**: Include all relevant information from the execution results
- **Maintain quality**: Ensure the output is professional and well-structured
- **Return raw JSON**: Do not wrap your response in code blocks or markdown formatting

## Example Output Structure
Your output should be a raw JSON object with keys matching the expected output names. For example:
{{
    "treatment_options": [
        "Option 1: Description of treatment option 1",
        "Option 2: Description of treatment option 2",
        "Option 3: Description of treatment option 3"
    ],
    "treatment_plan": "# Treatment Plan\\n\\n## Overview\\nComprehensive treatment plan based on diagnosis...\\n\\n## Implementation\\n1. Step 1\\n2. Step 2\\n\\n## Follow-up\\nRegular monitoring and adjustments as needed."
}}

## Context
- **Expected Outputs**: 
{expected_outputs}

- **Workflow Execution Results**: 
{workflow_execution_results}

Now, based on the workflow execution results, generate the appropriate structured output that matches the expected output format:

"""
