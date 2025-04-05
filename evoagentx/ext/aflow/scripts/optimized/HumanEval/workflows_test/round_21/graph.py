from typing import Literal
import evoagentx.workflow.operators as operator
from evoagentx.workflow.workflow_generator import WorkFlowGenerator
from evoagentx.models.base_model import BaseLLM
from evoagentx.models.model_utils import CostManager
import evoagentx.prompts.aflow_optimize_prompt as prompt
import asyncio

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
        self.error_log = []
        self.retry_attempts = 0
        self.retry_limit = 5  # Further increased retry limit for wider recovery possibilities
        self.retry_backoff = 2  # Increased backoff time for retries
        self.alternative_prompts = ["Suggest alternate strategies.", "Provide a different solution."]

    async def __call__(self, problem: str, entry_point: str):
        solution = await self.custom_code_generate.execute_async(input=problem, instruction=entry_point)
        testing_result = await self.test.execute_async(problem=problem, solution=solution['response'], entry_point=entry_point)
        if testing_result['result']:
            return solution['response'], float(self.cost_manager.total_cost) if isinstance(self.cost_manager.total_cost, (int, float, str)) else 0.0
        else:
            error_info = testing_result['solution']
            self.error_log.append(error_info)
            print(f"Error encountered: {error_info}")

            # Adding detailed error analysis for better understanding of error
            await self.custom.execute_async(input=f"Conduct a detailed analysis for error: {error_info}", instruction='Analyze and provide detailed insights on the error.')

            validation_result = await self.test.execute_async(problem=problem, solution=solution['response'], entry_point=entry_point)
            retry_solution = None
            
            if not validation_result['result']:
                await self.custom.execute_async(input=f"{error_info} for problem: {problem}", instruction='Log detailed error for analysis.')
                error_analysis = await self.custom.execute_async(input=problem + f" Error analysis: {error_info}", instruction='Analyze the nature of the error and suggest improvements.')
                print(f"Error analysis results: {error_analysis['response']}")
                error_response = await self.custom.execute_async(input=problem + f" Propose alternative solutions based on: {error_analysis['response']}", instruction=self.alternative_prompts[0])
                self.retry_attempts += 1
                print(f"Retry attempt: {self.retry_attempts} for problem: {problem}")
                if self.retry_attempts <= self.retry_limit:
                    await asyncio.sleep(self.retry_backoff)
                    retry_solution = await self.custom.execute_async(input=problem + f" Retrying after: {error_analysis['response']} or: {error_response['response']}", instruction=entry_point)
                    retry_testing_result = await self.test.execute_async(problem=problem, solution=retry_solution['response'], entry_point=entry_point)
                    if retry_testing_result['result']:
                        return retry_solution['response'], 0.0
                    else:
                        self.error_log.append(f"Retry failed: {retry_testing_result['solution']}. Error log updated.")
            # best_solution = await self.sc_ensemble.execute_async(solutions=[solution['response'], retry_solution['response']], problem=problem)
            # return best_solution['response'], 0.0
            # Only use ensemble when retry_solution exists
            if retry_solution:
                best_solution = await self.sc_ensemble.execute_async(solutions=[solution['response'], retry_solution['response']], problem=problem)
                return best_solution['response'], 0.0
            else:
                # If no retry was attempted or retry failed, return the original solution
                return solution['response'], 0.0
