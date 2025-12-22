# EvoPrompt Optimizer Tutorial

This tutorial will guide you through the process of setting up and running the EvoPrompt optimizer in EvoAgentX. We'll use the BIG-Bench Hard benchmark as an example to demonstrate how to optimize multi-agent workflow prompts using Genetic Algorithm (GA) and Differential Evolution (DE) algorithms.

## 1. Overview

The EvoPrompt optimizer in EvoAgentX enables you to:

- Automatically optimize prompts in multi-agent workflows using evolutionary algorithms
- Support both Genetic Algorithm (GA) and Differential Evolution (DE) optimization algorithms
- Evaluate optimization results on benchmark datasets
- Support parallel evolution and combination optimization of multi-node prompts
- Provide detailed training process visualization and logging

## 2. Performance Results

Based on our experimental results on the BIG-Bench Hard dataset, the EvoPrompt optimizer demonstrates significant performance improvements across multiple tasks:

### Performance Comparison Table

| Task | COT Baseline | GA Best | DE Best | GA Improvement | DE Improvement |
|------|-------------|---------|---------|----------------|----------------|
| **snarks** | 0.7109 | 0.8281 | 0.8281 | +16.5% | +16.5% |
| **geometric_shapes** | 0.3950 | 0.3700 | 0.4250 | -6.3% | +7.6% |
| **multistep_arithmetic_two** | 0.9450 | 0.9850 | 0.9750 | +4.2% | +3.2% |
| **ruin_names** | 0.5150 | 0.6850 | 0.7400 | +33.0% | +43.7% |

### Key Findings
- **ruin_names** task achieved the highest improvement, with DE algorithm reaching 43.7% improvement
- **snarks** task achieved identical and strong performance with both algorithms
- **multistep_arithmetic_two** despite having the highest baseline, still achieved consistent improvements
- Training process charts can be viewed in the corresponding `performance_summary_OVERALL.png` files

## 3. Setting Up the Environment

First, let's import the necessary modules for setting up the EvoPrompt optimizer:

```python
import asyncio
import os
import re
from collections import Counter
from dotenv import load_dotenv

from evoagentx.optimizers.evoprompt_optimizer import DEOptimizer, GAOptimizer
from evoagentx.benchmark.bigbenchhard import BIGBenchHard
from evoagentx.models import OpenAILLM, OpenAILLMConfig
from evoagentx.optimizers.engine.registry import ParamRegistry
from evoagentx.core.logging import logger
```

### Configure the LLM Models

You'll need a valid API key to initialize the LLM. See [Quickstart](../quickstart.md) for more details on how to set up your API key.

```python
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

# Evolution LLM configuration (for generating mutated prompts)
evo_llm_config = OpenAILLMConfig(
    model="gpt-4.1-nano",
    openai_key=OPENAI_API_KEY,
    stream=False,
    top_p=0.95,
    temperature=0.5  # Higher temperature for diverse prompt generation
)

# Evaluation LLM configuration (for task execution)
eval_llm_config = OpenAILLMConfig(
    model="gpt-4.1-nano",
    openai_key=OPENAI_API_KEY,
    stream=False,
    temperature=0  # Deterministic evaluation
)
llm = OpenAILLM(config=eval_llm_config)
```

## 4. Setting Up the Components

### Step 1: Define Program Workflow

The EvoPrompt optimizer requires a program class that defines the workflow logic. Here's an example using a multi-prompt voting mechanism:

```python
class SarcasmClassifierProgram:
    """
    A program using three-prompt majority voting ensemble for sarcasm classification.
    Each prompt serves as an independent "voter" that can evolve independently.
    """
    def __init__(self, model: OpenAILLM):
        self.model = model
        # Three different generic prompt nodes for diverse task handling
        self.prompt_direct = "As a straightforward responder, follow the task instruction exactly and provide the final answer."
        self.prompt_expert = "As an expert assistant, interpret the task instruction carefully and provide the final answer."
        self.prompt_cot = "As a thoughtful assistant, think step-by-step, then follow the task instruction and provide the final answer."
        self.task_instruction = "Respond with your final answer wrapped like this: FINAL_ANSWER(ANSWER)"

    def __call__(self, input: str) -> tuple[str, dict]:
        answers = []
        prompts = [self.prompt_direct, self.prompt_expert, self.prompt_cot]
        pattern = r"FINAL_ANSWER\((.*?)\)"

        for prompt in prompts:
            full_prompt = f"{prompt}\n\n{self.task_instruction}\n\nText:\n{input}"
            response = self.model.generate(prompt=full_prompt)
            prediction = response.content.strip()
            
            match = re.search(pattern, prediction, re.IGNORECASE)
            if match:
                answers.append(match.group(1))

        if not answers:
            return "N/A", {"votes": []}

        vote_counts = Counter(answers)
        most_common_answer = vote_counts.most_common(1)[0][0]
        
        return most_common_answer, {"votes": answers}

    def save(self, path: str):
        # Add saving logic here
        pass

    def load(self, path: str):
        # Add loading logic here
        pass
```

!!! note
    When defining your workflow, pay attention to the following key points:
    
    1. **Multi-prompt Nodes**: Each prompt variable defined in the program (like `prompt_direct`, `prompt_expert`, `prompt_cot`) can serve as an independent optimization node.
    
    2. **Voting Mechanism**: Using multi-prompt voting can improve result robustness, with each prompt node evolving independently.
    
    3. **Result Parsing**: Ensure the format returned by the `__call__` method is consistent with benchmark evaluation requirements.

### Step 2: Register Prompt Parameters

Use `ParamRegistry` to track the prompt nodes that need optimization:

```python
# Set up benchmark and program
benchmark = BIGBenchHard("snarks", dev_sample_num=15, seed=10)
program = SarcasmClassifierProgram(model=llm)

# Register prompt nodes
registry = ParamRegistry()
registry.track(program, "prompt_direct", name="direct_prompt_node")
registry.track(program, "prompt_expert", name="expert_prompt_node")
registry.track(program, "prompt_cot", name="cot_prompt_node")
```

### Step 3: Prepare the Benchmark

We use the BIG-Bench Hard benchmark, which contains various challenging tasks:

```python
# Available tasks include
tasks = [
    "snarks",                    # Sarcasm detection
    "geometric_shapes",          # Geometric shape recognition
    "multistep_arithmetic_two",  # Multi-step arithmetic
    "ruin_names",               # Corrupted name recognition
    "sports_understanding",      # Sports comprehension
    "logical_deduction_three_objects",  # Logical reasoning
    # ... more tasks
]

# Initialize benchmark
benchmark = BIGBenchHard(
    task_name="snarks", 
    dev_sample_num=15,  # Number of development set samples
    seed=10             # Random seed for reproducible results
)
```

## 5. Configuring and Running the EvoPrompt Optimizer

### Differential Evolution (DE) Optimizer

The DE algorithm optimizes prompts through differential mutation and crossover operations:

```python
# Configuration parameters
POPULATION_SIZE = 4           # Population size
ITERATIONS = 10              # Number of iterations
CONCURRENCY_LIMIT = 100      # Concurrency limit
COMBINATION_SAMPLE_SIZE = 3  # Sample size per combination

# DE Optimizer
optimizer_DE = DEOptimizer(
    registry=registry,
    program=program,
    population_size=POPULATION_SIZE,
    iterations=ITERATIONS,
    llm_config=evo_llm_config,
    concurrency_limit=CONCURRENCY_LIMIT,
    combination_sample_size=COMBINATION_SAMPLE_SIZE,
    enable_logging=True,         # Enable logging
    enable_early_stopping=True,  # Enable early stopping
    early_stopping_patience=3    # Early stopping patience
)

# Run optimization
logger.info("Optimizing with DE...")
await optimizer_DE.optimize(benchmark=benchmark)

# Evaluate results
logger.info("Evaluating with DE...")
de_metrics = await optimizer_DE.evaluate(benchmark=benchmark, eval_mode="test")
logger.info(f"DE results: {de_metrics['accuracy']}")
```

### Genetic Algorithm (GA) Optimizer

The GA algorithm optimizes prompts through selection, crossover, and mutation operations:

```python
# GA Optimizer
optimizer_GA = GAOptimizer(
    registry=registry,
    program=program,
    population_size=POPULATION_SIZE,
    iterations=ITERATIONS,
    llm_config=evo_llm_config,
    concurrency_limit=CONCURRENCY_LIMIT,
    combination_sample_size=COMBINATION_SAMPLE_SIZE,
    enable_logging=True,
    enable_early_stopping=True,
    early_stopping_patience=3
)

# Run optimization
logger.info("Optimizing with GA...")
await optimizer_GA.optimize(benchmark=benchmark)

# Evaluate results
logger.info("Evaluating with GA...")
ga_metrics = await optimizer_GA.evaluate(benchmark=benchmark, eval_mode="test")
logger.info(f"GA results: {ga_metrics['accuracy']}")
```

## 6. Optimization Parameters Explained

### Key Parameter Descriptions

- `population_size`: Population size, determines the number of individuals per generation
- `iterations`: Number of evolutionary iterations
- `combination_sample_size`: Number of samples for combination evaluation, used to reduce computational overhead
- `concurrency_limit`: Concurrency request limit, controls concurrent calls to LLM API
- `enable_early_stopping`: Whether to enable early stopping mechanism when performance stops improving
- `early_stopping_patience`: Early stopping patience rounds

### Algorithm Characteristics Comparison

| Algorithm | Characteristics | Applicable Scenarios |
|-----------|-----------------|---------------------|
| **DE** | Based on differential mutation, strong exploration capability | Complex optimization landscapes requiring global search |
| **GA** | Based on selection and crossover, stable convergence | Local optimization with known good solutions |

## 7. Logging and Visualization

The optimizer automatically generates detailed log files and visualization charts:

### Log File Structure
```
node_evolution_logs_{ALGO}_{MODEL}_{TASK}_{TIMESTAMP}/
├── combo_generation_XX_log.csv          # Combination evaluation logs per generation
├── evaluation_testset_test_*.csv         # Test set evaluation results
├── optimization_summary_{algo}.csv       # Optimization summary
├── best_config.json                      # Machine-readable best config (auto-saved)
├── performance_summary_OVERALL.png      # Training process visualization
└── individual_plots/                     # Individual node performance charts
    └── performance_plot_*.png
```

### Visualization Charts

The optimizer generates visualization charts of the training process, showing:
- Best and average performance per generation
- Evolution trajectories of individual prompt nodes
- Convergence curves of combination performance

## 8. Complete Example

Here's a complete running example:

```python
import asyncio
import os
from dotenv import load_dotenv
from evoagentx.optimizers.evoprompt_optimizer import DEOptimizer, GAOptimizer
from evoagentx.benchmark.bigbenchhard import BIGBenchHard
from evoagentx.models import OpenAILLM, OpenAILLMConfig
from evoagentx.optimizers.engine.registry import ParamRegistry

async def main():
    # Environment setup
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Configuration
    POPULATION_SIZE = 4
    ITERATIONS = 10
    CONCURRENCY_LIMIT = 100
    COMBINATION_SAMPLE_SIZE = 3
    DEV_SAMPLE_NUM = 15
    
    # LLM configuration
    evo_llm_config = OpenAILLMConfig(
        model="gpt-4.1-nano",
        openai_key=OPENAI_API_KEY,
        stream=False,
        top_p=0.95,
        temperature=0.5
    )
    
    eval_llm_config = OpenAILLMConfig(
        model="gpt-4.1-nano",
        openai_key=OPENAI_API_KEY,
        stream=False,
        temperature=0
    )
    llm = OpenAILLM(config=eval_llm_config)
    
    # Set up benchmark and program
    benchmark = BIGBenchHard("snarks", dev_sample_num=DEV_SAMPLE_NUM, seed=10)
    program = SarcasmClassifierProgram(model=llm)
    
    # Register prompt nodes
    registry = ParamRegistry()
    registry.track(program, "prompt_direct", name="direct_prompt_node")
    registry.track(program, "prompt_expert", name="expert_prompt_node")  
    registry.track(program, "prompt_cot", name="cot_prompt_node")
    
    # DE optimization
    optimizer_DE = DEOptimizer(
        registry=registry,
        program=program,
        population_size=POPULATION_SIZE,
        iterations=ITERATIONS,
        llm_config=evo_llm_config,
        concurrency_limit=CONCURRENCY_LIMIT,
        combination_sample_size=COMBINATION_SAMPLE_SIZE,
        enable_logging=True
    )
    
    await optimizer_DE.optimize(benchmark=benchmark)
    de_metrics = await optimizer_DE.evaluate(benchmark=benchmark, eval_mode="test")
    print(f"DE results: {de_metrics['accuracy']}")

if __name__ == "__main__":
    asyncio.run(main())
```

For a complete working example, please refer to [evoprompt_workflow.py](https://github.com/EvoAgentX/EvoAgentX/blob/main/examples/optimization/evoprompt/evoprompt_workflow.py).

## 9. Using the Optimized Program and Persisting JSON

- Optimized Program Object
  - After `optimize()` finishes, the optimizer automatically applies the best prompts back into your registered nodes. The `program` instance you passed in is already ready to use for downstream inference without extra steps.

- Auto-Saved JSON
  - Alongside CSV logs, the optimizer also saves `best_config.json` in the log directory. This contains a simple mapping of `{ node_name: optimized_prompt }`.

- Reload In A New Process/Instance
  - Option 1 (helper): `optimizer.load_and_apply_config("/path/to/best_config.json")`
  - Option 2 (registry): load JSON and call `registry.set(name, value)` for each entry.

Minimal examples:

```python
# Apply best_config.json to a fresh program via ParamRegistry
with open(json_path, "r", encoding="utf-8") as f:
    best_cfg = json.load(f)
for k, v in best_cfg.items():
    registry.set(k, v)
```

See examples:
- examples/optimization/evoprompt/evoprompt_bestconfig_json.py
- examples/optimization/evoprompt/evoprompt_save_load_json_min.py

## 10. Best Practices

1. **Set reasonable population size**: Recommend 4-8 individuals, balancing exploration and computational overhead
2. **Use early stopping**: Avoid overfitting and save computational resources
3. **Adjust temperature parameters**: Use higher temperature (0.5-0.8) for evolution, lower temperature (0-0.2) for evaluation
4. **Monitor logs**: Pay attention to convergence trends and adjust parameters timely
5. **Multi-task testing**: Validate optimizer generality across multiple tasks

## 11. Troubleshooting

### Common Issues

**Q: Performance doesn't improve during optimization?**
A: Check baseline performance, adjust temperature parameters, increase population size or iterations.

**Q: High memory usage?**
A: Reduce `concurrency_limit` and `combination_sample_size`.

**Q: API rate limit issues?**
A: Lower `concurrency_limit`, add appropriate delays.

For more optimizer examples and advanced configurations, please check other scripts in the [examples/optimization/evoprompt](https://github.com/EvoAgentX/EvoAgentX/tree/main/examples/optimization/evoprompt) directory.
