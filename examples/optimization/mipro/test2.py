import numpy as np
from evoagentx.workflow.operators import Predictor
import os
import json 
from dotenv import load_dotenv
from typing import Any, Tuple

from evoagentx.benchmark import MATH
from evoagentx.core.logging import logger
from evoagentx.models import OpenAILLM, OpenAILLMConfig
from evoagentx.optimizers import MiproOptimizer
from evoagentx.core.callbacks import suppress_logger_info
from evoagentx.utils.mipro_utils.register_utils import MiproRegistry
from evoagentx.workflow.operators import Predictor



load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
def sample_uniform_01(n=10):
    print(np.random.uniform(low=0.0, high=1.0, size=n))

openai_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY, stream=True, output_response=False)
executor_llm = OpenAILLM(config=openai_config)
optimizer_config = OpenAILLMConfig(model="gpt-4o", openai_key=OPENAI_API_KEY, stream=True, output_response=False)
optimizer_llm = OpenAILLM(config=optimizer_config)

predictor = Predictor(llm = executor_llm)
response = predictor.execute(question="'A rectangular piece of paper measures 4 units by 5 units. Several lines are drawn parallel to the edges of the paper, going from one edge to the other. A rectangle determined by the intersections of some of these lines is called basic if\n\n(i) all four sides of the rectangle are segments of drawn line segments, and\n(ii) no segments of drawn lines lie inside the rectangle.\n\nGiven that the total length of all lines drawn is exactly 2007 units, let $N$ be the maximum possible number of basic rectangles determined.  Find $N$.")
print(response['answer'])
print("\n\n\n")
print(response['reasoning'])