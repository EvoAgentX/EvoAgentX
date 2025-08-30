import os
from dotenv import load_dotenv
from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.agents import CustomizeAgent
from utils import create_deep_copy, save_block_config, load_block_config

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class Debate:
    """Debate agent，用于评估多个解决方案并选择最佳方案"""
    
    def __init__(self, llm: OpenAILLM):
        self.llm = llm
        self.agent = CustomizeAgent(
            name="Debate",
            description="Evaluate multiple solutions and select the best one",
            prompt="""You are a Debate agent. Your task is to examine multiple solutions to a problem and select the best one.

Problem: {problem}
{context}
Solutions: {solutions}

Please examine each solution carefully and provide your reasoning for selecting the best one in the following format:

## reasoning
Your detailed analysis of each solution and why you chose the best one

## index
The index of the solution you choose (0-based)

## answer
Your refined version of the chosen solution""",
            llm_config=llm.config,
            inputs=[
                {"name": "problem", "type": "string", "description": "The problem to solve"},
                {"name": "context", "type": "string", "description": "Additional context information", "required": False},
                {"name": "solutions", "type": "string", "description": "List of solutions to evaluate"}
            ],
            outputs=[
                {"name": "reasoning", "type": "string", "description": "Your reasoning for the selection"},
                {"name": "index", "type": "string", "description": "Index of the chosen solution"},
                {"name": "answer", "type": "string", "description": "Your refined answer"}
            ],
            parse_mode="title"
        )
    
    def __call__(self, problem: str, solutions: list, **kwargs):
        """兼容原Debate的调用方式"""
        response = self.execute(problem, solutions, **kwargs)
        return response['answer'], {
            "problem": problem, 
            "solutions": solutions,
            "reasoning": response['reasoning'], 
            "index": response['index'],
            "answer": response['answer']
        }
    
    def execute(self, problem: str, solutions: list, **kwargs) -> dict:
        """执行辩论任务"""
        context = kwargs.pop('context', None)
        
        # 格式化solutions
        solutions_text = "\n".join([f"{i}. {sol}" for i, sol in enumerate(solutions)])
        
        inputs = {"problem": problem, "solutions": solutions_text}
        if context:
            inputs["context"] = context
        
        response = self.agent(inputs=inputs)
        return {
            'reasoning': response.content.reasoning,
            'index': response.content.index,
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
def create_debate_agent(llm: OpenAILLM, **kwargs):
    """创建DebateAgent的便捷函数"""
    return Debate(llm, **kwargs)


