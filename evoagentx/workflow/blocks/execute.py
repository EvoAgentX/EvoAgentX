import json
from evoagentx.workflow.blocks.block import block
from evoagentx.workflow.operators import CodeReflector

class execute(block):
    def __init__(self, predictor, benchmark, llm) -> None:        
        self.benchmark = benchmark
        self.predictor = predictor
        self.codereflector = CodeReflector(llm=llm)
        self.search_space = [0,1]

    def __call__(self, problem, **kwargs):
        
        test_cases = kwargs.pop("testcases", None)


        predictor_prediction = self.predictor.execute(problem = problem, **kwargs)
        
        answer = predictor_prediction['answer']

        if test_cases:
            traceback = self.benchmark.evaluate(answer, test_cases)
        else:       
            traceback = "This dataset does not provide test cases"

        code_reflector_prediction = self.codereflector.execute(problem = problem, previous_solution = answer, traceback = traceback)

        return code_reflector_prediction['answer'], {"problem":problem, 
                                                     "test_cases":test_cases,
                                                     "reasoning":code_reflector_prediction["reasoning"], 
                                                     "correctness":code_reflector_prediction["correctness"], 
                                                     "answer":code_reflector_prediction["answer"],
                                                     "predictor_reasoning":predictor_prediction["reasoning"],
                                                     "predictor_answer":predictor_prediction["answer"]}

    def execute(self, problem, solution, **kwargs):
        test_cases = kwargs.get('test_cases', None)
        self.benchmark.trace_back = True

        for _ in range(self.n):
            if test_cases:
                traceback = self.benchmark.evaluate(solution, test_cases)
            else:
                traceback = "This dataset does not provide test cases"
                
            code_reflector_prediction = self.codereflector.execute(problem = problem, previous_solution = solution, traceback = traceback)
            solution = code_reflector_prediction["answer"]
            
        return solution
    
    def get_registry(self):
        return ['executer.codereflector.prompt']
    
    def save(self, path: str):
        params = {
            "codereflector": self.codereflector.prompt,
            "predictor": self.predictor.prompt
        }
        
        with open(path, "w") as f:
            json.dump(params, f)
    
    def load(self, path: str):
        with open(path, "r") as f:
            params = json.load(f)
            self.code_reflector.prompt = params["codereflector"]
            self.predictor.prompt = params["predictor"]