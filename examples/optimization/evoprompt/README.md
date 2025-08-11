# Node-wise Evolution Examples

This directory contains examples demonstrating the new **node-wise evolution architecture** for prompt optimization in EvoAgentX. This architecture represents a significant improvement over traditional mega-prompt evolution approaches.

## 🎯 What is Node-wise Evolution?

Instead of evolving one large "mega-prompt" containing all prompts concatenated together, node-wise evolution treats each registered prompt as an independent node that evolves separately. The system then evaluates different combinations of evolved prompts to determine the fitness of each individual prompt.

### Key Benefits

- **🧠 Reduced LLM Complexity**: No complex XML parsing or mega-prompt handling
- **🎯 Independent Evolution**: Each prompt evolves based on its own performance
- **💰 Cost Control**: Combination sampling controls evaluation overhead
- **📊 Fine-grained Tracking**: Performance metrics per individual prompt node
- **🔄 Scalability**: Easily add more prompt nodes without architectural changes

## 📁 Examples Overview

### 1. `quick_start_node_evolution.py` - Beginner Friendly
**Perfect for getting started with node-wise evolution**

```python
# Simple 2-node system
system_prompt = "You are an expert classifier..."
task_prompt = "Classify the text as (A) or (B)..."

# Register as separate nodes
registry.track(program, "system_prompt", name="system_node")
registry.track(program, "task_prompt", name="task_node")

# Each node evolves independently!
```

**Features:**
- 2 prompt nodes (system + task)
- 3 variants per node = 3×3 = 9 total combinations
- Samples 5 combinations for evaluation
- GA optimization with 2 evolution rounds

### 2. `bbh_snarks_node_evolution_example.py` - Comprehensive Demo
**Shows both GA and DE algorithms with detailed logging**

```python
# 3-node ensemble system
prompt_direct = "From the two sentences provided..."
prompt_expert = "You are an expert in linguistics..."  
prompt_cot = "Consider the context and potential meanings..."

# Each prompt is an independent voter
registry.track(program, "prompt_direct", name="direct_prompt_node")
registry.track(program, "prompt_expert", name="expert_prompt_node")
registry.track(program, "prompt_cot", name="cot_prompt_node")
```

**Features:**
- 3 prompt nodes in ensemble voting system
- 4 variants per node = 4³ = 64 total combinations  
- Samples 8 combinations for evaluation
- Compares DE vs GA performance
- Detailed performance tracking per node per generation

### 3. `advanced_node_evolution_examples.py` - Real-world Applications
**Demonstrates complex multi-stage and ensemble systems**

#### Multi-Stage QA System
```python
# 3-stage processing pipeline
analysis_prompt = "Analyze the question carefully..."
reasoning_prompt = "Think through the logical steps..."
answer_prompt = "Provide a clear and accurate answer..."

# Each stage evolves independently
registry.track(program, "analysis_prompt", name="analysis_stage")
registry.track(program, "reasoning_prompt", name="reasoning_stage") 
registry.track(program, "answer_prompt", name="answer_stage")
```

#### Ensemble Voting System
```python
# 3 different voting strategies
direct_vote = "Look at both options..."
expert_vote = "As an expert in this domain..."
analytical_vote = "Systematically analyze each option..."

# Each strategy evolves separately
registry.track(program, "direct_vote", name="direct_strategy")
registry.track(program, "expert_vote", name="expert_strategy")
registry.track(program, "analytical_vote", name="analytical_strategy")
```

## 🚀 Quick Start

1. **Install Dependencies**
```bash
pip install evoagentx python-dotenv
```

2. **Set Environment Variable**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

3. **Run Quick Start Example**
```bash
python quick_start_node_evolution.py
```

## 🔧 Configuration Options

### Population Size Control
```python
population_size = 4  # Each node will have 4 prompt variants
```

### Combination Sampling
```python
# Control evaluation cost vs accuracy trade-off
combination_sample_size = 10  # Sample 10 out of total combinations

# Examples:
# 3 nodes × 4 variants = 4³ = 64 total combinations
# combination_sample_size = 10 → evaluate only 10 combinations
# combination_sample_size = None → evaluate all 64 combinations
```

### Algorithm Selection
```python
# Choose between Genetic Algorithm and Differential Evolution
optimizer = GAOptimizer(...)  # Genetic Algorithm
# or
optimizer = DEOptimizer(...)  # Differential Evolution
```

## 📊 Understanding the Output

The evolution process provides detailed tracking:

```
📈 Performance Evolution by Node:

Generation 1:
  system_node: Best=0.750, Avg=0.680
  task_node: Best=0.720, Avg=0.650

Generation 2:  
  system_node: Best=0.820, Avg=0.750
  task_node: Best=0.780, Avg=0.720
```

Each node's performance is tracked independently, allowing you to see which parts of your system are improving most.

## 🏗️ Architecture Comparison

### Old Approach: Mega-prompt Evolution
```python
mega_prompt = """
<system_prompt>You are an expert...</system_prompt>
<task_prompt>Classify the text...</task_prompt>
"""
# LLM must parse and evolve complex XML structure
```

### New Approach: Node-wise Evolution
```python
# Each prompt evolves independently
system_prompt = "You are an expert..."
task_prompt = "Classify the text..."

# Clean, focused evolution per prompt
registry.track(program, "system_prompt", name="system_node")
registry.track(program, "task_prompt", name="task_node")
```

## 💡 Best Practices

1. **Start Small**: Begin with 2-3 nodes to understand the system
2. **Control Costs**: Use combination sampling for systems with many nodes
3. **Track Performance**: Monitor per-node metrics to identify bottlenecks
4. **Iterate**: Start with small population sizes and fewer iterations for faster experimentation
5. **Choose Algorithms**: Try both GA and DE to see which works better for your use case

## 🔬 Advanced Usage

### Custom Combination Sampling Strategy
```python
# You can implement custom sampling logic in the optimizer
def custom_sample_combinations(all_combinations):
    # Your custom sampling logic here
    return sampled_combinations
```

### Dynamic Population Sizes
```python
# Different nodes can theoretically have different population sizes
# (Future enhancement)
```

### Performance Analysis
```python
# Access detailed performance data
best_scores_per_gen = optimizer.best_scores_per_gen
avg_scores_per_gen = optimizer.avg_scores_per_gen

# Analyze evolution trajectories per node
for gen_name in best_scores_per_gen:
    for node_name in best_scores_per_gen[gen_name]:
        print(f"{gen_name} {node_name}: {best_scores_per_gen[gen_name][node_name]}")
```

## 🤝 Contributing

Feel free to contribute additional examples or improvements:
- Multi-modal prompt systems
- Dynamic combination sampling strategies  
- Performance visualization tools
- Integration with other benchmarks

## 📚 References

- [EvoPrompt Paper](https://arxiv.org/abs/2302.14838) - Original prompt evolution concept
- [EvoAgentX Documentation](../../docs/) - Full framework documentation
- [Genetic Algorithms](https://en.wikipedia.org/wiki/Genetic_algorithm) - GA background
- [Differential Evolution](https://en.wikipedia.org/wiki/Differential_evolution) - DE background
