import json
from evoagentx.workflow.blocks.block import block
from evoagentx.workflow.operators import Summarizer

class summarize(block):
    def __init__(
            self,
            predictor,
            llm,
    ):
        self.summarizer = Summarizer(llm=llm)
        self.predictor = predictor
        self.search_space = [0, 1, 2, 3, 4]

    def __call__(self, problem, **kwargs):
        """将 summarize 作为独立模块使用，返回最终的预测答案"""
        context = kwargs.pop('context', None)

        if context is not None:
            summary = self.summarizer.execute(problem=problem, context=context)
            context = summary['summary']
        
        prediction = self.predictor.execute(problem=problem, context=context)
        
        return prediction['answer'], {"problem": problem, 
                                      "summary":context, 
                                      "reasoning":prediction['reasoning'], 
                                      "answer":prediction['answer']}

    def execute(self, problem, **kwargs):
        """将 summarize 作为 workflow 组件使用，返回总结后的上下文"""
        context = kwargs.pop('context', None)
        if context is not None:
            for _ in range(self.n):
                summary = self.summarizer.execute(problem=problem, context=context)
                context = summary['summary']
        
        return context

    def save(self, path: str):
        params = {"summarizer": self.summarizer.prompt, "predictor": self.predictor.prompt}

        with open(path, "w") as f:
            json.dump(params, f)
    
    def load(self, path: str):
        with open(path, "r") as f:
            params = json.load(f)
            self.summarizer.prompt = params["summarizer"]
            self.predictor.prompt = params["predictor"]
    
    def get_registry(self):
        return ["summarizer.summarizer.prompt"]