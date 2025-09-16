import os
from dotenv import load_dotenv
from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.agents import CustomizeAgent
from utils import create_deep_copy, save_block_config, load_block_config

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class Reflect:
    """Reflect agent，用于反思和评估答案的正确性"""
    
    def __init__(self, llm: OpenAILLM):
        self.llm = llm
        self.agent = CustomizeAgent(
            name="Reflect",
            description="Reflect on the correctness of an answer",
            prompt="""You are a Reflect agent. Your task is to review an answer and criticize where it might be wrong. If you are absolutely sure it is correct, output 'True' in 'correctness'.

Question: {problem}
{context}
Text: {text}

Please analyze the answer carefully and provide your reasoning in the following format:

## reasoning
Your detailed reasoning about the correctness of the answer

## feedback
Your feedback on what might be wrong or what could be improved

## correctness
True/False indicating if the answer is correct given the question""",
            llm_config=llm.config,
            inputs=[
                {"name": "problem", "type": "string", "description": "The original problem"},
                {"name": "context", "type": "string", "description": "Additional context information", "required": False},
                {"name": "text", "type": "string", "description": "The answer to evaluate"}
            ],
            outputs=[
                {"name": "reasoning", "type": "string", "description": "Your reasoning about correctness"},
                {"name": "feedback", "type": "string", "description": "Your feedback on the answer"},
                {"name": "correctness", "type": "string", "description": "True/False indicating correctness"}
            ],
            parse_mode="title"
        )
    
    def __call__(self, problem: str, text: str, **kwargs):
        """兼容原Reflect的调用方式"""
        response = self.execute(problem, text, **kwargs)
        return response['correctness'].lower() == 'true', {
            "problem": problem, 
            "text": text,
            "reasoning": response['reasoning'], 
            "feedback": response['feedback'],
            "correctness": response['correctness']
        }
    
    def execute(self, problem: str, text: str, **kwargs) -> dict:
        """执行反思任务"""
        context = kwargs.pop('context', None)
        
        inputs = {"problem": problem, "text": text}
        if context:
            inputs["context"] = context
        
        response = self.agent(inputs=inputs)
        return {
            'reasoning': response.content.reasoning,
            'feedback': response.content.feedback,
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
def create_reflect_agent(llm: OpenAILLM, **kwargs):
    """创建ReflectAgent的便捷函数"""
    return Reflect(llm, **kwargs)


