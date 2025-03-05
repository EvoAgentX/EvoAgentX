from pydantic import Field
from typing import Any, Optional, Callable
import re

from ..core.module import BaseModule
from ..core.registry import MODEL_REGISTRY
from ..models.model_configs import LLMConfig
from .operators import AnswerGenerate, ScEnsemble
from ..models.base_model import LLMOutputParser
from pydantic import BaseModel
from ..core.logging import logger


class ActionGraph(BaseModule):

    name: str = Field(description="The name of the ActionGraph.")
    description: str = Field(description="The description of the ActionGraph.")
    llm_config: LLMConfig = Field(description="The config of LLM used to execute the ActionGraph.")

    def init_module(self):
        if self.llm_config:
            llm_cls = MODEL_REGISTRY.get_model(self.llm_config.llm_type)
            self._llm = llm_cls(config=self.llm_config)
    
    def __call__(self, *args: Any, **kwargs: Any) -> dict:
        return self.execute(*args, **kwargs)
    
    def execute(self, *args, **kwargs) -> dict:
        raise NotImplementedError(f"The execute function for {type(self).__name__} is not implemented!")
    

class QAActionGraph(ActionGraph):

    def __init__(self, llm_config: LLMConfig, **kwargs):

        name = kwargs.pop("name") if "name" in kwargs else "Simple QA Workflow"
        description = kwargs.pop("description") if "description" in kwargs else \
            "This is a simple QA workflow that use self-consistency to make predictions."
        super().__init__(name=name, description=description, llm_config=llm_config, **kwargs)
        self.answer_generate = AnswerGenerate(self._llm)
        self.sc_ensemble = ScEnsemble(self._llm)
        
    def execute(self, problem: str) -> dict:

        solutions = [] 
        for _ in range(3):
            response = self.answer_generate(input=problem)
            answer = response["answer"]
            solutions.append(answer)
        ensemble_result = self.sc_ensemble(solutions=solutions)
        best_answer = ensemble_result["response"]
        return {"answer": best_answer}
    
    
class GraphOptimize(LLMOutputParser):
    modification: str = Field(default="", description="modification")
    graph: str = Field(default="", description="graph")
    prompt: str = Field(default="", description="prompt")
    
    @staticmethod
    def _parse_xml_content(content: str) -> dict:
        """Parse XML content into a dictionary."""
        data = {}
        # Extract modification
        modification_match = re.search(r'<modification>(.*?)</modification>', content, re.DOTALL)
        if modification_match:
            data['modification'] = modification_match.group(1).strip()
            
        # Extract graph
        graph_match = re.search(r'<graph>(.*?)</graph>', content, re.DOTALL)
        if graph_match:
            data['graph'] = graph_match.group(1).strip()
            
        # Extract prompt
        prompt_match = re.search(r'<prompt>(.*?)</prompt>', content, re.DOTALL)
        if prompt_match:
            data['prompt'] = prompt_match.group(1).strip()
            
        return data
    
    @classmethod
    def parse(cls, content: str, parse_mode: str = "xml", parse_func: Optional[Callable] = None, **kwargs):
        """Parse the content into a GraphOptimize object."""
        if parse_mode == "xml":
            data = cls._parse_xml_content(content)
        else:
            data = cls.get_content_data(content=content, parse_mode=parse_mode, parse_func=parse_func, **kwargs)
        data.update({"content": content})
        return cls(**data)
    
    
class HumanEvalActionGraph(ActionGraph):
    
    def __init__(self, llm_config: LLMConfig, **kwargs):
        
        name = kwargs.pop("name") if "name" in kwargs else "HumanEval Action Graph"
        description = kwargs.pop("description") if "description" in kwargs else \
            "This is a HumanEval action graph that use self-consistency to make predictions."
        super().__init__(name=name, description=description, llm_config=llm_config, **kwargs)
        
    def execute(self, problem: str) -> dict:
        
        response = self._llm.generate(prompt=problem, parser=GraphOptimize, parse_mode="xml")
        # logger.info(f"modification is {response.modification}")
        # logger.info(f"graph is {response.graph}")
        # logger.info(f"prompt is {response.prompt}")
        
        return {
            "modification": response.modification,
            "graph": response.graph,
            "prompt": response.prompt
        }
        
        
    
    
    
    