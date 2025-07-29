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
        "database_connection_url": "",
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
- You should generate a workflow with given inputs and outputs formats.

## Requirement documentation:
{requirement}

"""



import os
from dotenv import load_dotenv
from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.workflow import WorkFlowGraph, WorkFlowGenerator
from evoagentx.core.module_utils import parse_json_from_llm_output
from .requirement_analysis_prompt import REQUIREMENT_ANALYSIS_PROMPT

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
module_save_path = "server/workflow_{workflow_name}.json"


default_llm_config = {
    "model": "gpt-4o-mini",
    "openai_key": OPENAI_API_KEY,
    "stream": True,
    "output_response": True,
    "max_tokens": 16000
}
llm = OpenAILLM(OpenAILLMConfig(**default_llm_config))

def main():
    overall_requirement = """写一个宠物管理网站，需要有ai工作流完成疾病原因分析和治疗推荐"""
    
    llm_response = llm.generate(prompt = overall_requirement, system_message=REQUIREMENT_ANALYSIS_PROMPT)
    
    # Format the prompt with the actual requirement
    total_requirements = WORKFLOW_REQUIREMENT_PROMPT.format(requirement=llm_response.content)
    
    # Actually call the LLM with the prompt
    llm_response = llm.generate(total_requirements)
    from pdb import set_trace; set_trace()
    
    # Parse the JSON from the LLM's response
    workflow_requirement = parse_json_from_llm_output(llm_response.content)
    
    print("LLM Response:")
    print(llm_response)
    print("\n" + "="*50 + "\n")
    print("Parsed Workflow Requirement:")
    print(workflow_requirement)
    
    for workflow in workflow_requirement["workflows"]:
        print(workflow["workflow_name"])
        print(workflow["workflow_requirement"])
        print(workflow["workflow_inputs"])
        print(workflow["workflow_outputs"])
        print("\n" + "="*50 + "\n") 
        
        ## _______________ Workflow Creation _______________
        tools = []
        wf_generator = WorkFlowGenerator(llm=llm, tools=tools)
        workflow_graph: WorkFlowGraph = wf_generator.generate_workflow(goal=WORKFLOW_GENERATION_GOAL_PROMPT.format(workflow_inputs=workflow["workflow_inputs"], workflow_outputs=workflow["workflow_outputs"], requirement=workflow["workflow_requirement"]))
        workflow_graph.save_module(module_save_path.format(workflow_name=workflow["workflow_name"]))

if __name__ == "__main__":
    main()

