import os
from dotenv import load_dotenv
from evoagentx.agents import AgentManager
from evoagentx.models import OpenAILLM, OpenAILLMConfig
from evoagentx.benchmark import MATH
from evoagentx.workflow import SequentialWorkFlowGraph
from evoagentx.core.callbacks import suppress_logger_info
from evoagentx.evaluators import Evaluator
from evoagentx.core.logging import logger
from evoagentx.prompts import GEPAPromptTemplate
from evoagentx.optimizers.gepa_optimizer import WorkFlowGEPAOptimizer

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class MathSplits(MATH):

    def _load_data(self):
        # load the original test data
        super()._load_data()
        # split the data into dev and test
        import numpy as np
        np.random.seed(42)
        permutation = np.random.permutation(len(self._test_data))
        full_test_data = self._test_data
        # randomly select 100 samples for training and 100 samples for test
        self._train_data = [full_test_data[idx] for idx in permutation[:100]]
        self._test_data = [full_test_data[idx] for idx in permutation[100:200]]
        # populate dev split so downstream evaluators have examples
        self._dev_data = list(self._train_data)

    def get_input_keys(self):
        return ["problem"]


def collate_func(example: dict) -> dict:
    return {"problem": example["problem"]}


# Custom evaluator with natural language feedback for GEPA
# Users must provide domain-specific feedback explaining:
#   - What went wrong (specific errors)
#   - Why it went wrong (reasoning flaws)
#   - How to fix it (reference solutions, recommendations)

class MathGEPAEvaluator:
    """
    Custom evaluator for MATH with detailed natural language feedback.

    Provides domain-specific feedback explaining errors, reasoning flaws,
    and recommendations for improvement.
    """

    def __init__(self, benchmark: MathSplits):
        self.benchmark = benchmark

    def metric(self, example, prediction, trace=None, pred_name=None, pred_trace=None):
        """
        GEPA-compatible metric returning score with natural language feedback.

        Returns:
            dspy.Prediction with score and feedback
        """
        import dspy
        import re

        # Extract answers
        try:
            # Get correct answer from example
            correct_answer = self.benchmark.get_label(example)

            # Extract model's answer from prediction
            if hasattr(prediction, 'solution'):
                model_output = prediction.solution
            elif hasattr(prediction, 'answer'):
                model_output = prediction.answer
            else:
                model_output = str(prediction)

            # Try to extract boxed answer
            boxed_pattern = r'\\boxed\{([^}]+)\}'
            model_matches = re.findall(boxed_pattern, model_output)
            model_answer = model_matches[-1] if model_matches else model_output.strip()

            # Normalize answers for comparison
            def normalize(s):
                return s.strip().replace(' ', '').replace('$', '').lower()

            is_correct = normalize(model_answer) == normalize(correct_answer)
            score = 1.0 if is_correct else 0.0

        except Exception as e:
            score = 0.0
            is_correct = False
            model_answer = "ERROR"
            correct_answer = self.benchmark.get_label(example) if hasattr(example, 'answer') else "Unknown"

        # Generate detailed natural language feedback for GEPA
        feedback_parts = []

        if is_correct:
            feedback_parts.append(f"✓ CORRECT! The answer {model_answer} matches the expected answer.")
            feedback_parts.append("The reasoning approach was sound and led to the correct solution.")
        else:
            feedback_parts.append(f"✗ INCORRECT. Expected: {correct_answer}, but got: {model_answer}")
            feedback_parts.append("\nError Analysis:")

            # Provide specific feedback based on the problem
            if hasattr(example, 'problem'):
                problem = example.problem if isinstance(example.problem, str) else str(example.problem)
                feedback_parts.append(f"Problem: {problem[:200]}...")  # First 200 chars

            # Add reference solution if available
            if hasattr(example, 'solution'):
                feedback_parts.append("\n📚 REFERENCE SOLUTION:")
                feedback_parts.append(example.solution[:500])  # First 500 chars
                feedback_parts.append("\nKey Insight: Study this solution to understand the correct approach.")

            # Provide actionable guidance
            feedback_parts.append("\n💡 RECOMMENDATIONS:")
            feedback_parts.append("1. Break down the problem into smaller, manageable steps")
            feedback_parts.append("2. Verify each calculation or logical step before proceeding")
            feedback_parts.append("3. Ensure the final answer is in the correct format (\\boxed{})")
            feedback_parts.append("4. Double-check your work against the problem requirements")

            # Add problem-specific hints based on type
            if 'algebra' in str(example).lower():
                feedback_parts.append("5. For algebra problems: Check variable substitutions and equation solving steps")
            elif 'geometry' in str(example).lower():
                feedback_parts.append("5. For geometry problems: Verify angle calculations and apply relevant theorems")
            elif 'calculus' in str(example).lower():
                feedback_parts.append("5. For calculus problems: Review differentiation/integration rules carefully")

        feedback = "\n".join(feedback_parts)

        return dspy.Prediction(score=score, feedback=feedback)

    def __call__(self, program, evalset, **kwargs):
        """Evaluate program on dataset."""
        import dspy
        from tqdm import tqdm

        results = []
        for example in tqdm(evalset, desc="Evaluating"):
            try:
                # Run program
                prediction = program(**example.inputs())

                # Get score and feedback
                result = self.metric(example, prediction)
                score = result.score if isinstance(result, dspy.Prediction) else result

                results.append(score)
            except Exception as e:
                print(f"Error evaluating example: {e}")
                results.append(0.0)

        avg_score = sum(results) / len(results) if results else 0.0
        return avg_score * 100  # Return as percentage


math_graph_data = {
    "goal": r"Answer the math question. The answer should be in box format, e.g., \boxed{{123}}.",
    "tasks": [
        {
            "name": "answer_generate",
            "description": "Answer generation for Math.",
            "inputs": [
                {"name": "problem", "type": "str", "required": True, "description": "The problem to solve."}
            ],
            "outputs": [
                {"name": "solution", "type": "str", "required": True, "description": "The generated answer."}
            ],
            "prompt_template": GEPAPromptTemplate(
                instruction=r"Let's think step by step to answer the math question.",
            ),
            "parse_mode": "title"
        }
    ]
}


def main():

    # Configure task model (student) - fast, cost-efficient model
    task_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY, stream=True, output_response=False)
    task_llm = OpenAILLM(config=task_config)

    # Configure reflection model - stronger model for analyzing failures
    reflection_config = OpenAILLMConfig(model="gpt-4o", openai_key=OPENAI_API_KEY, stream=True, output_response=False)
    reflection_llm = OpenAILLM(config=reflection_config)

    benchmark = MathSplits()
    workflow_graph: SequentialWorkFlowGraph = SequentialWorkFlowGraph.from_dict(math_graph_data)
    agent_manager = AgentManager()
    agent_manager.add_agents_from_workflow(workflow_graph, llm_config=task_config)

    # Create custom evaluator with domain-specific feedback
    logger.info("Creating GEPA evaluator with natural language feedback...")
    gepa_evaluator = MathGEPAEvaluator(benchmark=benchmark)

    # We use the standard Evaluator for workflow execution, but override
    # the metric with our custom feedback-generating function
    workflow_evaluator = Evaluator(
        llm=task_llm,
        agent_manager=agent_manager,
        collate_func=collate_func,
        num_workers=20,
        verbose=True
    )

    # Define the GEPA optimizer
    logger.info("Initializing GEPA optimizer...")
    logger.info("GEPA will use detailed natural language feedback to guide prompt evolution")
    optimizer = WorkFlowGEPAOptimizer(
        graph=workflow_graph,
        evaluator=workflow_evaluator,     # Workflow evaluator for execution
        optimizer_llm=task_llm,           # Student model for task execution
        reflection_llm=reflection_llm,    # Strong model for reflective prompt evolution
        eval_rounds=1,
        auto="medium",                    # Auto configuration: n=12, val_size=300
        reflection_minibatch_size=16,     # Number of examples per reflection step
        candidate_selection_strategy='pareto',  # Use Pareto frontier
        component_selector='round_robin',       # Optimize one component per iteration
        use_merge=True,                         # Enable variant merging
        max_merge_invocations=5,                # Maximum merge operations
        skip_perfect_score=True,                # Skip perfect examples
        num_threads=20,                         # Parallel evaluation
        track_stats=True,                       # Track optimization statistics
        save_path="examples/output/gepa/math_gepa",
    )

    # Override the metric with our GEPA-compatible feedback generator
    # This ensures GEPA receives rich natural language feedback
    optimizer.metric = gepa_evaluator.metric
    logger.info("✓ Custom metric with natural language feedback configured")

    logger.info("Starting GEPA optimization...")
    logger.info("GEPA will use reflective prompt evolution to optimize instructions")
    logger.info("Unlike MIPRO, GEPA focuses on instruction-only optimization (no few-shot examples)")
    logger.info("Reflection LM will analyze failures and propose improved prompts\n")

    optimizer.optimize(dataset=benchmark)

    logger.info("\nRestoring best program...")
    optimizer.restore_best_program()

    logger.info("Evaluating optimized program on test set...")
    with suppress_logger_info():
        test_score = optimizer.evaluate(dataset=benchmark, eval_mode="test")
    logger.info(f"\nTest set performance (after GEPA optimization): {test_score}%")

    logger.info("\n" + "="*60)
    logger.info("GEPA Optimization Complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
