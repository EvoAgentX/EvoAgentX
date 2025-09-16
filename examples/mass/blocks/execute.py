import os
from dotenv import load_dotenv
from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.agents import CustomizeAgent
from utils import create_deep_copy, save_block_config, load_block_config

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class Execute:
    """Execute agent，用于执行代码和测试"""
    
    def __init__(self, llm: OpenAILLM):
        self.llm = llm
        self.agent = CustomizeAgent(
            name="Execute",
            description="Execute code and provide test results",
            prompt="""You are an Execute agent. Your task is to execute code and provide test results with traceback information.

Question: {question}
Solution: {solution}

Please execute the solution and provide the results in the following format:

## traceback
The execution traceback including test cases, execution results, and ground truth. If there is an error, include the relevant traceback.

## correctness
True/False based on the correctness of execution feedback. If there is an error message, output 'False'.""",
            llm_config=llm.config,
            inputs=[
                {"name": "question", "type": "string", "description": "The original question"},
                {"name": "solution", "type": "string", "description": "The code solution to execute"}
            ],
            outputs=[
                {"name": "traceback", "type": "string", "description": "Execution traceback and test results"},
                {"name": "correctness", "type": "string", "description": "True/False indicating correctness"}
            ],
            parse_mode="title"
        )
    
    def __call__(self, question: str, solution: str, **kwargs):
        """兼容原Execute的调用方式"""
        response = self.execute(question, solution, **kwargs)
        return response['correctness'].lower() == 'true', {
            "question": question, 
            "solution": solution,
            "traceback": response['traceback'], 
            "correctness": response['correctness']
        }
    
    def execute(self, question: str, solution: str, **kwargs) -> dict:
        """执行代码任务"""
        inputs = {"question": question, "solution": solution}
        
        response = self.agent(inputs=inputs)
        return {
            'traceback': response.content.traceback,
            'correctness': response.content.correctness
        }
    
    def save(self, path: str):
        """保存agent配置"""
        return self.agent.save_module(path)
    
    @classmethod
    def load(cls, path: str, llm: OpenAILLM):
        """加载agent配置"""
        agent_data = CustomizeAgent.load_module(path, llm_config=llm.config)
        instance = cls(llm)
        instance.agent = CustomizeAgent(**agent_data)
        return instance

# 便捷创建函数
def create_execute_agent(llm: OpenAILLM, **kwargs):
    """创建ExecuteAgent的便捷函数"""
    return Execute(llm, **kwargs)


