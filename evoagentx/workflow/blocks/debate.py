import json
from typing import Any
from evoagentx.workflow.blocks.block import block
from evoagentx.workflow.operators import Debater

class debate(block):
    def __init__(self,
                 predictor,
                 llm):
        self.debater = Debater(llm=llm)
        self.predictor = predictor
        self.search_space = [0,1,2,3,4]

    def __call__(self, problem, **kwargs) -> Any:
        context = kwargs.get('context', None)
        predictions = []
        for _ in range(2):
            prediction = self.predictor.execute(problem=problem, **kwargs)
            predictions.append(prediction)

        debater_prediction = self.debater.execute(problem=problem, solutions=[prediction['answer'] for prediction in predictions], context=context)
    
        return debater_prediction['answer'], {"problem": problem, 
                                              "reasoning": debater_prediction['reasoning'],
                                              "answer": debater_prediction['answer'],
                                              "predictor_reasoning": predictions[0]['reasoning'],
                                              "predictor_answer": predictions[0]['answer']}
        

    def execute(self, problem, solutions, **kwargs):
        context = kwargs.get('context', None)
        for i in range(self.n):
            prediction = self.debater.execute(problem=problem, solutions=solutions, context=context)
            print(f"debate round {i}: {prediction}")
            index = int(prediction['index'])
            if not (0 <= index < len(solutions)):
                index = len(solutions) - 1
            solutions[index] = prediction['answer']
            # 否则跳过

        return prediction['answer']
    
    def save(self, path: str):
        params = {
            "debater": self.debater.prompt,
            "predictor": self.predictor.prompt
        }
        
        with open(path, "w") as f:
            json.dump(params, f)
    
    def load(self, path: str):
        with open(path, "r") as f:
            params = json.load(f)
            self.debater.prompt = params["debater"]
            self.predictor.prompt = params["predictor"]
    
    def get_registry(self):
        return ["debater.debater.prompt"]