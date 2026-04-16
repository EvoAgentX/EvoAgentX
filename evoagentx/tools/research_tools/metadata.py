from pydantic import Field, BaseModel
from typing import Dict, Any, List, Optional

from ...core.decorators import atomic_method
from ...models import LLMOutputParser


# mock ToolMetadata and ToolResult to avoid duplicate cost tracking 
class ResearchToolMetadata(BaseModel):
    tool_name: str
    cost_breakdown: Dict[str, float] = Field(default_factory=dict)

    @atomic_method
    def add_cost_breakdown(self, cost_breakdown: Dict[str, float]):
        for key, value in cost_breakdown.items():
            self.cost_breakdown[key] = self.cost_breakdown.get(key, 0.0) + value 


class ResearchToolResult(BaseModel):
    metadata: ResearchToolMetadata
    result: Any 


class PaperMetadataOutputParser(LLMOutputParser):

    title: Optional[str] = Field(default=None, description="The title of the paper.")
    authors: Optional[List[str]] = Field(default_factory=list, description="The full names of all the authors.")
    abstract: Optional[str] = Field(default=None, description="The abstract of the paper.")
