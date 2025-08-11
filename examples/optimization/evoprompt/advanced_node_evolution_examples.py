"""
Advanced Example: Multi-Node RAG System Evolution

This example shows how to evolve a complex multi-prompt RAG (Retrieval-Augmented Generation) 
system where different prompts handle different aspects of the task:
- Document retrieval prompt
- Information synthesis prompt  
- Answer generation prompt

Each prompt evolves independently while the system is evaluated as a whole.
"""

import asyncio
import os
from typing import Dict, List
from dotenv import load_dotenv

from evoagentx.optimizers.evo2_optimizer import DEOptimizer
from evoagentx.benchmark.bigbenchhard import BIGBenchHard
from evoagentx.models import OpenAILLM, OpenAILLMConfig
from evoagentx.optimizers.engine.registry import ParamRegistry
from evoagentx.core.logging import logger

class MultiStageQASystem:
    """
    A multi-stage QA system with three specialized prompts:
    1. Analysis prompt: Analyzes the question
    2. Reasoning prompt: Performs logical reasoning
    3. Answer prompt: Generates the final answer
    """
    
    def __init__(self, model: OpenAILLM):
        self.model = model
        
        # Three specialized prompts that will evolve independently
        self.analysis_prompt = "Analyze the following question carefully. Identify the key information needed to answer it."
        
        self.reasoning_prompt = "Based on the analysis, think through the logical steps needed to solve this problem. Consider all relevant factors."
        
        self.answer_prompt = "Given the analysis and reasoning above, provide a clear and accurate answer. Choose (A) or (B)."

    def __call__(self, input: str) -> tuple[str, dict]:
        """
        Multi-stage processing:
        1. Analyze the question
        2. Perform reasoning
        3. Generate final answer
        """
        # Stage 1: Analysis
        analysis_input = f"{self.analysis_prompt}\n\nQuestion: {input}"
        analysis_response = self.model.generate(prompt=analysis_input)
        analysis_result = analysis_response.content.strip()
        
        # Stage 2: Reasoning
        reasoning_input = f"{self.reasoning_prompt}\n\nQuestion: {input}\nAnalysis: {analysis_result}"
        reasoning_response = self.model.generate(prompt=reasoning_input)
        reasoning_result = reasoning_response.content.strip()
        
        # Stage 3: Final Answer
        answer_input = f"{self.answer_prompt}\n\nQuestion: {input}\nAnalysis: {analysis_result}\nReasoning: {reasoning_result}"
        answer_response = self.model.generate(prompt=answer_input)
        final_answer = answer_response.content.strip()
        
        # Extract (A) or (B) from final answer
        if "(A)" in final_answer.upper():
            prediction = "(A)"
        elif "(B)" in final_answer.upper():
            prediction = "(B)"
        else:
            prediction = "(A)"  # Default fallback
            
        return prediction, {
            "analysis": analysis_result,
            "reasoning": reasoning_result, 
            "final_answer": final_answer
        }

class EnsembleClassifier:
    """
    An ensemble classifier that uses multiple voting strategies.
    Each strategy represents a different node for evolution.
    """
    
    def __init__(self, model: OpenAILLM):
        self.model = model
        
        # Different voting strategies as separate nodes
        self.direct_vote = "Look at both options (A) and (B). Which one is correct? Answer with (A) or (B)."
        
        self.expert_vote = "As an expert in this domain, carefully evaluate options (A) and (B). Which is the better choice? Respond (A) or (B)."
        
        self.analytical_vote = "Systematically analyze each option. Consider pros and cons. Which option (A) or (B) is more logical?"

    def __call__(self, input: str) -> tuple[str, dict]:
        """Run ensemble voting with three different strategies."""
        votes = []
        vote_details = {}
        
        strategies = [
            ("direct", self.direct_vote),
            ("expert", self.expert_vote), 
            ("analytical", self.analytical_vote)
        ]
        
        for strategy_name, prompt in strategies:
            full_prompt = f"{prompt}\n\nQuestion: {input}"
            response = self.model.generate(prompt=full_prompt)
            result = response.content.strip()
            
            vote_details[strategy_name] = result
            
            # Extract vote
            if "(A)" in result.upper():
                votes.append("(A)")
            elif "(B)" in result.upper():
                votes.append("(B)")
        
        # Majority voting
        if not votes:
            final_vote = "(A)"
        else:
            vote_count = {"(A)": votes.count("(A)"), "(B)": votes.count("(B)")}
            final_vote = max(vote_count.keys(), key=lambda x: vote_count[x])
        
        return final_vote, {
            "votes": votes,
            "details": vote_details,
            "final_vote": final_vote
        }

async def run_multistage_evolution():
    """Demonstrate evolution of a multi-stage QA system."""
    
    logger.info("🚀 Multi-Stage QA System Evolution")
    logger.info("="*50)
    
    # Setup
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("Please set OPENAI_API_KEY")

    llm_config = OpenAILLMConfig(
        model="gpt-4o-mini",
        openai_key=OPENAI_API_KEY,
        stream=False
    )
    llm = OpenAILLM(config=llm_config)

    # Small benchmark for demo
    benchmark = BIGBenchHard("snarks", sample_num=15)
    benchmark._load_data()

    # Create multi-stage system
    program = MultiStageQASystem(model=llm)
    
    logger.info("📝 Initial prompts:")
    logger.info(f"  Analysis: '{program.analysis_prompt[:50]}...'")
    logger.info(f"  Reasoning: '{program.reasoning_prompt[:50]}...'")
    logger.info(f"  Answer: '{program.answer_prompt[:50]}...'")

    # Register each stage as a separate node
    registry = ParamRegistry()
    registry.track(program, "analysis_prompt", name="analysis_stage")
    registry.track(program, "reasoning_prompt", name="reasoning_stage")
    registry.track(program, "answer_prompt", name="answer_stage")

    # Setup optimizer with controlled combination sampling
    optimizer = DEOptimizer(
        registry=registry,
        program=program,
        population_size=4,          # 4 variants per stage
        iterations=3,               # 3 evolution rounds
        llm_config=llm_config,
        concurrency_limit=30,
        combination_sample_size=50  # Sample 10 out of 4^3=64 combinations
    )
    
    logger.info(f"🔬 Evolution setup:")
    logger.info(f"  - Stages: 3 (analysis, reasoning, answer)")
    logger.info(f"  - Population per stage: 4")
    logger.info(f"  - Total combinations: 4³ = 64")
    logger.info(f"  - Sampled combinations: 10")

    # Run evolution
    logger.info("\n🧬 Starting evolution...")
    best_config, best_scores, avg_scores = await optimizer.optimize(benchmark=benchmark)

    # Show results
    logger.info("\n🎉 Evolution completed!")
    
    for node_name, evolved_prompt in best_config.items():
        logger.info(f"\n📝 {node_name}:")
        logger.info(f"  '{evolved_prompt[:80]}...'")
    
    return best_config

async def run_ensemble_evolution():
    """Demonstrate evolution of an ensemble voting system."""
    
    logger.info("\n" + "="*50)
    logger.info("🚀 Ensemble Voting System Evolution")
    logger.info("="*50)
    
    # Setup (reuse config from above)
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    llm_config = OpenAILLMConfig(
        model="gpt-4o-mini",
        openai_key=OPENAI_API_KEY,
        stream=False
    )
    llm = OpenAILLM(config=llm_config)

    benchmark = BIGBenchHard("snarks", sample_num=15)
    benchmark._load_data()

    # Create ensemble system
    program = EnsembleClassifier(model=llm)
    
    logger.info("📝 Initial voting strategies:")
    logger.info(f"  Direct: '{program.direct_vote[:40]}...'")
    logger.info(f"  Expert: '{program.expert_vote[:40]}...'")
    logger.info(f"  Analytical: '{program.analytical_vote[:40]}...'")

    # Register each voting strategy as a separate node
    registry = ParamRegistry()
    registry.track(program, "direct_vote", name="direct_strategy")
    registry.track(program, "expert_vote", name="expert_strategy")
    registry.track(program, "analytical_vote", name="analytical_strategy")

    # Use GA this time for variety
    from evoagentx.optimizers.evo2_optimizer import GAOptimizer
    
    optimizer = GAOptimizer(
        registry=registry,
        program=program,
        population_size=3,          # 3 variants per strategy
        iterations=2,               # 2 evolution rounds
        llm_config=llm_config,
        concurrency_limit=25,
        combination_sample_size=6   # Sample 6 out of 3^3=27 combinations
    )
    
    logger.info(f"🔬 GA Evolution setup:")
    logger.info(f"  - Strategies: 3 (direct, expert, analytical)")
    logger.info(f"  - Population per strategy: 3")
    logger.info(f"  - Total combinations: 3³ = 27")
    logger.info(f"  - Sampled combinations: 6")

    # Run evolution
    logger.info("\n🧬 Starting GA evolution...")
    best_config, best_scores, avg_scores = await optimizer.optimize(benchmark=benchmark)

    # Show results
    logger.info("\n🎉 GA Evolution completed!")
    
    for node_name, evolved_prompt in best_config.items():
        logger.info(f"\n📝 {node_name}:")
        logger.info(f"  '{evolved_prompt[:80]}...'")
    
    return best_config

async def main():
    """Run both examples to showcase different use cases."""
    
    logger.info("🌟 Advanced Node-wise Evolution Examples")
    logger.info("="*60)
    
    # Example 1: Multi-stage QA system
    multistage_config = await run_multistage_evolution()
    
    # Example 2: Ensemble voting system  
    ensemble_config = await run_ensemble_evolution()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("🏆 EXAMPLES SUMMARY")
    logger.info("="*60)
    
    logger.info("\n✨ Key Achievements:")
    logger.info("  1. 🎯 Multi-stage system: Each processing stage evolved independently")
    logger.info("  2. 🗳️  Ensemble system: Different voting strategies optimized separately")
    logger.info("  3. 💰 Cost control: Combination sampling reduced evaluation overhead")
    logger.info("  4. 🧠 Simplicity: No complex XML parsing or mega-prompt handling")
    logger.info("  5. 📊 Granularity: Fine-grained performance tracking per node")
    
    logger.info("\n🔧 Architecture Benefits:")
    logger.info("  • Each prompt node has its own evolution trajectory")
    logger.info("  • Controllable computational cost via sampling")
    logger.info("  • Better LLM prompt clarity and focus")
    logger.info("  • Scalable to systems with many prompt components")
    logger.info("  • Flexible algorithm choice per optimization run")

if __name__ == "__main__":
    asyncio.run(main())
