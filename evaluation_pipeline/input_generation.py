import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict

from evoagentx.agents import CustomizeAgent
from evoagentx.models import BaseLLM

INPUT_GENERATION_PROMPT = """
You are a specialized assistant that generates valid pseudo-inputs for a workflow.

**Task:** Given a JSON array of input requirements, generate a single JSON object of pseudo-inputs.

**Instructions:**
- Populate all `required: true` fields with a sensible value based on their `type` and `description`.
- For `required: false` fields, you have a choice:
    - You may either populate the field with a realistic, valid value.
    - Or, you may set the value to `null`.
- The output must be a single JSON object.

**Example:**

## Input Requirements
```json
[
    {{
        "name": "prompt",
        "description": "User's story idea or prompt",
        "type": "string",
        "required": true
    }},
    {{
        "name": "genre",
        "description": "Story genre (romance, sci-fi, mystery, etc.)",
        "type": "string",
        "required": true
    }},
    {{
        "name": "style",
        "description": "Writing style preference",
        "type": "string",
        "required": false
    }},
    {{
        "name": "length",
        "description": "Story length (short, medium, long)",
        "type": "string",
        "required": true
    }},
    {{
        "name": "characters",
        "description": "Character descriptions or names",
        "type": "array",
        "required": false
    }},
    {{
        "name": "setting",
        "description": "Story setting or location",
        "type": "string",
        "required": false
    }}
]


## Generated Inputs
```json
{{
    "prompt": "A young detective investigates a haunted lighthouse.",
    "genre": "mystery",
    "style": "noir",
    "length": "short",
    "characters": ["Eleanor Vance", "Captain Thorne"],
    "setting": null
}}
```

**Begin:**

## Input Requirements
```json
{input_requirements}
```

## Generated Inputs
```json

"""


def generate_workflow_inputs(workflow_inputs: dict, llm: BaseLLM) -> dict:
    workflow_inputs_str = json.dumps(workflow_inputs, indent=4, ensure_ascii=False)
    output_parser = CustomizeAgent.create_action_output(workflow_inputs, "output_parser")

    prompt = INPUT_GENERATION_PROMPT.format(input_requirements=workflow_inputs_str)
    llm_output = llm.generate(
        prompt=prompt,
        parser=output_parser,
        parse_mode="json"
    )

    return llm_output.get_structured_data()


def scan_workflow_generation_files(directory_path: str) -> List[str]:
    """扫描workflow_generation_eval_data目录中的所有JSON文件"""
    directory = Path(directory_path)
    if not directory.exists():
        raise FileNotFoundError(f"目录不存在: {directory_path}")
    
    json_files = list(directory.glob("*.json"))
    return [str(file) for file in json_files]


def load_workflow_data(file_path: str) -> Dict:
    """加载workflow JSON文件并提取相关字段"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return {
        "workflow_name": data.get("workflow_name", ""),
        "workflow_id": data.get("workflow_id", ""),
        "workflow_requirement": data.get("workflow_requirement", ""),
        "workflow_inputs": data.get("workflow_inputs", []),
        "workflow_outputs": data.get("workflow_outputs", [])
    }


def generate_execution_filename(gen_filename: str) -> str:
    """将workflow_gen_X.json转换为workflow_exe_X.json"""
    filename = Path(gen_filename).stem
    if filename.startswith("workflow_gen_"):
        number = filename.replace("workflow_gen_", "")
        return f"workflow_exe_{number}.json"
    else:
        # 如果不符合预期格式，就简单替换
        return filename.replace("_gen_", "_exe_") + ".json"


def process_all_workflows(
    generation_dir: str = "evaluation_pipeline/workflow_generation_eval_data",
    execution_dir: str = "evaluation_pipeline/workflow_execution_eval_data",
    llm: BaseLLM = None
) -> None:
    """处理所有workflow文件，生成测试数据"""
    if llm is None:
        from evoagentx.models import OpenAILLM, OpenAILLMConfig
        llm_config = OpenAILLMConfig(model="gpt-4o", output_response=True)
        llm = OpenAILLM(llm_config)
    
    # 确保输出目录存在
    Path(execution_dir).mkdir(parents=True, exist_ok=True)
    
    # 扫描所有JSON文件
    json_files = scan_workflow_generation_files(generation_dir)
    print(f"找到 {len(json_files)} 个workflow文件")
    
    for file_path in json_files:
        try:
            print(f"正在处理: {file_path}")
            
            # 加载workflow数据
            workflow_data = load_workflow_data(file_path)
            
            # 生成测试输入数据
            test_inputs = generate_workflow_inputs(workflow_data["workflow_inputs"], llm)
            
            # 创建执行测试数据结构
            execution_data = {
                "workflow_name": workflow_data["workflow_name"],
                "workflow_id": workflow_data["workflow_id"],
                "workflow_requirement": workflow_data["workflow_requirement"],
                "test_inputs": test_inputs,
                "expected_outputs": workflow_data["workflow_outputs"],
                "metadata": {
                    "generated_from": Path(file_path).name,
                    "generation_timestamp": datetime.now().isoformat()
                }
            }
            
            # 生成输出文件名
            output_filename = generate_execution_filename(Path(file_path).name)
            output_path = Path(execution_dir) / output_filename
            
            # 保存到文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(execution_data, f, indent=4, ensure_ascii=False)
            
            print(f"✓ 已生成: {output_path}")
            
        except Exception as e:
            print(f"✗ 处理文件 {file_path} 时出错: {e}")


if __name__ == "__main__":
    # 使用示例
    process_all_workflows()
    print("所有workflow测试数据生成完成！")