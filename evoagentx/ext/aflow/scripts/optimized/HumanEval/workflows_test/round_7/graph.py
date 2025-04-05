from typing import Literal
import evoagentx.workflow.operators as operator
from evoagentx.workflow.workflow_generator import WorkFlowGenerator
from evoagentx.models.base_model import BaseLLM
from evoagentx.models.model_utils import CostManager
import evoagentx.prompts.aflow_optimize_prompt as prompt

DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP"]

class AFlowWorkflowGenerator(WorkFlowGenerator):
    
    def __init__(self, *, 
                 name: str,
                 llm: BaseLLM,
                 dataset: DatasetType,
                 **kwargs):
        super().__init__(name=name, llm=llm, **kwargs)
        
        self.llm = llm
        self.dataset = dataset
        self.name = name
        self.cost_manager = CostManager()
        self.custom = operator.Custom(self.llm)
        self.custom_code_generate = operator.CustomCodeGenerate(self.llm)
        self.answer_generate = operator.AnswerGenerate(self.llm)
        self.test = operator.Test(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)
        self.error_log = []  # Initialize error log

    async def __call__(self, problem: str, entry_point: str):
        # Generate initial solution using custom code generation
        solution = await self.custom_code_generate.execute_async(input=problem, instruction=entry_point)
        # Test the generated solution
        testing_result = await self.test.execute_async(problem=problem, solution=solution['response'], entry_point=entry_point)
        if testing_result['result']:  # If test passes
            return solution['response'], float(self.cost_manager.total_cost) if isinstance(self.cost_manager.total_cost, (int, float, str)) else 0.0
        else:  # If test fails
            # Log error information and analyze the type of error
            error_info = testing_result['solution']  # Capturing error information from test
            self.error_log.append(error_info)  # Store error information for further analysis
            print(f"Error encountered: {error_info}")
            # Adding a detailed error analysis before retrying
            error_analysis = await self.custom.execute_async(input=problem + f" Error analysis: {error_info}", instruction='Analyze the nature of the error and suggest improvements.')
            print(f"Error analysis results: {error_analysis['response']}")
            # Retry with improvements based on error analysis
            retry_solution = await self.custom.execute_async(input=problem + f" Retrying after: {error_analysis['response']}", instruction=entry_point)
            # Use ensemble to select the final solution amid errors
            best_solution = await self.sc_ensemble.execute_async(solutions=[solution['response'], retry_solution['response']], problem=problem)
            return best_solution['response'], 0.0
