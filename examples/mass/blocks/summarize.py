import os
from dotenv import load_dotenv
from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.agents import CustomizeAgent
from utils import create_deep_copy, save_block_config, load_block_config

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class Summarize:
    """Summarize agent，用于总结和提取相关信息"""
    
    def __init__(self, llm: OpenAILLM):
        self.llm = llm
        self.agent = CustomizeAgent(
            name="Summarize",
            description="Summarize relevant information from context",
            prompt="""You are a Summarize agent. Your task is to retrieve relevant information from context that is ONLY helpful in answering the question. Include all key information. Do not repeat context.

Question: {problem}
Context: {context}

Please extract and summarize only the information that is relevant to answering the question in the following format:

## summary
Your concise summary of relevant information from the context""",
            llm_config=llm.config,
            inputs=[
                {"name": "problem", "type": "string", "description": "The question to answer"},
                {"name": "context", "type": "string", "description": "The context to summarize"}
            ],
            outputs=[
                {"name": "summary", "type": "string", "description": "Summary of relevant information"}
            ],
            parse_mode="title"
        )
    
    def __call__(self, problem: str, context: str, **kwargs):
        """兼容原Summarize的调用方式"""
        response = self.execute(problem, context, **kwargs)
        return response['summary'], {
            "problem": problem, 
            "context": context,
            "summary": response['summary']
        }
    
    def execute(self, problem: str, context: str, **kwargs) -> dict:
        """执行总结任务"""
        inputs = {"problem": problem, "context": context}
        
        response = self.agent(inputs=inputs)
        return {
            'summary': response.content.summary
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
def create_summarize_agent(llm: OpenAILLM, **kwargs):
    """创建SummarizeAgent的便捷函数"""
    return Summarize(llm, **kwargs)


