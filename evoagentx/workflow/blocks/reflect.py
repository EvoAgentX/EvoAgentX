import json
from evoagentx.workflow.blocks.block import block
from evoagentx.workflow.operators import Reflector, Refiner

class reflect(block):
    def __init__(
            self,
            predictor,
            llm,

    ):
        self.predictor = predictor
        self.reflector = Reflector(llm=llm)
        self.refiner = Refiner(llm=llm)
        self.search_space = [0, 1, 2, 3, 4]

    def __call__(self, problem, **kwargs):
        """将 reflect 作为独立模块使用，返回最终精炼后的答案"""
        context = kwargs.pop('context', None)
        
        # 首先生成初始预测
        predictor_prediction = self.predictor.execute(problem=problem, **kwargs)
        answer = predictor_prediction['answer']
        
        reflector_prediction = self.reflector.execute(problem=problem, 
                                                        text=answer,
                                                        context=context)
        refiner_prediction = self.refiner.execute(problem=problem, 
                                                    previous_answer=answer,
                                                    reflection=reflector_prediction['feedback'],
                                                    correctness=reflector_prediction['correctness'],
                                                    context=context)
            
        answer = refiner_prediction['answer']
        
        return answer, {"problem": problem, 
                        "context": context, 
                        "reflector_reasoning": reflector_prediction['reasoning'],
                        "reflector_feedback": reflector_prediction['feedback'], 
                        "reflector_correctness": reflector_prediction['correctness'],
                        "refiner_reasoning": refiner_prediction['reasoning'], 
                        "refiner_answer": refiner_prediction['answer'],
                        "predictor_reasoning": predictor_prediction['reasoning'],
                        "predictor_answer": predictor_prediction['answer']}

    def execute(self, problem, solution, **kwargs):
        """将 reflect 作为 workflow 组件使用，对给定的 solution 进行反思和精炼"""
        context = kwargs.get('context', None)
        current_solution = solution
        
        for i in range(self.n):
            reflector_prediction = self.reflector.execute(problem=problem,
                                                          text=current_solution,
                                                          context=context)
            
            refiner_prediction = self.refiner.execute(problem=problem,
                                                      previous_answer=current_solution,
                                                      reflection=reflector_prediction['feedback'],
                                                      correctness=reflector_prediction['correctness'],
                                                      context=context)
            current_solution = refiner_prediction['answer']
        
        return current_solution

    def save(self, path: str):
        params = {
            "predictor": self.predictor.prompt, 
            "reflector": self.reflector.prompt,
            "refiner": self.refiner.prompt
        }
        
        with open(path, "w") as f:
            json.dump(params, f)
    
    def load(self, path: str):
        with open(path, "r") as f:
            params = json.load(f)
            self.predictor.prompt = params["predictor"]
            self.reflector.prompt = params["reflector"]
            self.refiner.prompt = params["refiner"]
    
    def get_registry(self):
        return ["reflector.reflector.prompt", "reflector.refiner.prompt"]