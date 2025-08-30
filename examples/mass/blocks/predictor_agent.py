import os
from dotenv import load_dotenv
from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.agents import CustomizeAgent

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class Predictor:
    """CustomizeAgent实现的Predictor，对应原operators.py中的Predictor"""
    
    def __init__(self, llm: OpenAILLM):
        self.llm = llm
        self.agent = CustomizeAgent(
            name="Predictor",
            description="Predict the answer to the problem",
            prompt="""You are a Predictor agent. Your task is to analyze the given problem and provide a well-reasoned prediction.

Problem: {problem}
{context}

Please analyze the problem carefully and provide your reasoning and prediction in the following format:

## reasoning
Your detailed reasoning process here

## answer
Your final prediction/answer here""",
            llm_config=llm.config,
            inputs=[
                {"name": "problem", "type": "string", "description": "The problem to solve"},
                {"name": "context", "type": "string", "description": "Additional context information", "required": False}
            ],
            outputs=[
                {"name": "reasoning", "type": "string", "description": "Your reasoning for this problem"},
                {"name": "answer", "type": "string", "description": "Your prediction for this problem"}
            ],
            parse_mode="title"
        )
    
    def __call__(self, problem, **kwargs):
        """兼容原Predictor的调用方式"""
        response = self.execute(problem, **kwargs)
        return response['answer'], {"problem": problem, "reasoning": response['reasoning'], "answer": response['answer']}
    
    def execute(self, problem, **kwargs) -> dict:
        """执行预测任务"""
        context = kwargs.pop('context', None)
        inputs = {"problem": problem}
        if context:
            inputs["context"] = context
        
        response = self.agent(inputs=inputs)
        return {
            'reasoning': response.content.reasoning,
            'answer': response.content.answer
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
def create_predictor_agent(llm: OpenAILLM, **kwargs):
    """创建PredictorAgent的便捷函数"""
    return Predictor(llm, **kwargs)


    
