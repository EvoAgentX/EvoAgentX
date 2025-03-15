import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))  # 添加项目根目录到 PYTHONPATH

from evoagentx.models.siliconflow_model import SiliconFlow
from evoagentx.models.model_configs import SiliconFlowConfig

def test_siliconflow_model(siliconflow_key: str, 
                           model: str,
                           question: str,
                           stream: bool):
    config = SiliconFlowConfig(siliconflow_key=siliconflow_key,
                           model=model)

    model = SiliconFlow(config=config)

    model.init_model()

    model.single_generate(messages=[{'role': 'user', 
                                    'content': question}],
                        stream=stream)
    
    print(f"model.get_cost(): {model.get_cost()}")
    
   
    
if __name__ == "__main__":
    
    siliconflow_key = "sk-vhxzxtorlkllodmmcthkhjiidjvnkkgpaliydwrdkrqzilpw" # ruihong'key
    model = "deepseek-ai/DeepSeek-V3"
    question = "Hello, who are you?"
    stream = False
    
    test_siliconflow_model(siliconflow_key=siliconflow_key,
                           model=model,
                           question=question,
                           stream=stream)