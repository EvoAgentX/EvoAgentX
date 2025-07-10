# Acknowledgement: Modified from OpenEvolve (https://github.com/codelion/openevolve/blob/main/openevolve/controller.py) under Apache-2.0 license

import os
import random
import time
import uuid
from typing import Coroutine, List, Optional

import numpy as np
from pydantic import Field
from tqdm import tqdm

from ..core.logging import logger
from ..core.module import BaseModule
from ..evaluators.alphaevolve_evaluator import AlphaEvolveEvaluator
from ..models.base_model import BaseLLM
from ..utils.alphaevolve_utils.code_utils import (
    apply_diff,
    extract_code_language,
    extract_diffs,
    format_diff_summary,
    parse_full_rewrite,
)
from ..utils.alphaevolve_utils.format_utils import (
    format_improvement_safe,
    format_metrics_safe,
)
from .alphaevolve.config import AlphaEvolveDatabaseConfig, AlphaEvolvePromptConfig
from .alphaevolve.database import AlphaEvolveDatabase, Program
from .alphaevolve.ensemble import LLMEnsemble
from .alphaevolve.sampler import PromptSampler


class AlphaEvolveOptimizer(BaseModule):
    initial_program_path: str = Field(description="The path to the initial program.")
    llms: List[BaseLLM] = Field(description="The LLMs to use for the evolution process.")
    weights: Optional[List[float]] = Field(default=None, description="The weights for the LLMs. If not specified, equal weights are used.")
    prompt_config: AlphaEvolvePromptConfig = Field(default=AlphaEvolvePromptConfig(), description="The prompt configuration for the evolution process.")
    database_config: AlphaEvolveDatabaseConfig = Field(default=AlphaEvolveDatabaseConfig(), description="The database configuration for the evolution process.")
    evaluator: AlphaEvolveEvaluator = Field(description="The evaluator for the evolution process.")
    max_iterations: int = Field(default=10000, description="The maximum number of iterations.")
    target_score: Optional[float] = Field(default=None, description="The target score to reach (continues until reached if specified)")
    checkpoint_interval: int = Field(default=100, description="The interval at which to checkpoint the program.")
    diff_based_evolution: bool = Field(default=True, description="Whether to use diff-based evolution.")
    allow_full_rewrites: bool = Field(default=False, description="Whether to allow full rewrites.")
    max_code_length: int = Field(default=10000, description="The maximum length of the code.")
    output_dir: Optional[str] = Field(default=None, description="The output directory.")
    seed: Optional[int] = Field(default=None, description="The random seed for reproducibility.")


    def init_module(self):
        # Set up output directory
        self.output_dir = self.output_dir or os.path.join(
            os.path.dirname(self.initial_program_path), "alphaevolve_output"
        )
        os.makedirs(self.output_dir, exist_ok=True)

        # Set random seed for reproducibility if specified
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        # Load initial program
        with open(self.initial_program_path, "r") as f:
            self.initial_program_code = f.read()
        self.language = extract_code_language(self.initial_program_code)

        # Extract file extension from initial program
        self.file_extension = os.path.splitext(self.initial_program_path)[1]
        if not self.file_extension:
            # Default to .py if no extension found
            self.file_extension = ".py"
        else:
            # Make sure it starts with a dot
            if not self.file_extension.startswith("."):
                self.file_extension = f".{self.file_extension}"

        # Initialize components
        self.llm_ensemble = LLMEnsemble(self.llms, self.weights)
        self.prompt_sampler = PromptSampler(self.prompt_config)

        # Pass random seed to database if specified
        if self.seed is not None:
            self.database_config.random_seed = self.seed

        self.database = AlphaEvolveDatabase(self.database_config)

        logger.remove(0)
        logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)


    async def run(self) -> Coroutine[None, None, Program | None]:
        """
        Run the evolution process

        Returns:
            Best program found
        """

        # Define start_iteration before creating the initial program
        start_iteration = self.database.last_iteration

        # Only add initial program if starting fresh (not resuming from checkpoint)
        # Check if we're resuming AND no program matches initial code to avoid pollution
        should_add_initial = (
            start_iteration == 0
            and len(self.database.programs) == 0
            and not any(
                p.code == self.initial_program_code for p in self.database.programs.values()
            )
        )

        if should_add_initial:
            logger.info("Adding initial program to database")
            initial_program_id = str(uuid.uuid4())

            # Evaluate the initial program
            initial_metrics = await self.evaluator.evaluate_program(
                self.initial_program_code, initial_program_id
            )

            initial_program = Program(
                id=initial_program_id,
                code=self.initial_program_code,
                language=self.language,
                metrics=initial_metrics,
                iteration_found=start_iteration,
            )

            self.database.add(program=initial_program)
        else:
            logger.info(
                f"Skipping initial program addition (resuming from iteration {start_iteration} with {len(self.database.programs)} existing programs)"
            )

        # Main evolution loop
        total_iterations = start_iteration + self.max_iterations

        logger.info(
            f"Starting evolution from iteration {start_iteration} for {self.max_iterations} iterations (total: {total_iterations})"
        )

        # Island-based evolution variables
        programs_per_island = max(
            1, self.max_iterations // (self.database_config.num_islands * 10)
        )  # Dynamic allocation
        current_island_counter = 0

        logger.info(f"Using island-based evolution with {self.database_config.num_islands} islands")
        self.database.log_island_status()


        for i in tqdm(range(start_iteration, total_iterations)):
            iteration_start = time.time()

            # Manage island evolution - switch islands periodically
            if i > start_iteration and current_island_counter >= programs_per_island:
                self.database.next_island()
                current_island_counter = 0
                logger.debug(f"Switched to island {self.database.current_island}")

            current_island_counter += 1

            # Sample parent and inspirations from current island
            parent, inspirations = self.database.sample()

            # Get artifacts for the parent program if available
            parent_artifacts = self.database.get_artifacts(parent.id)

            # Build prompt
            prompt = self.prompt_sampler.build_prompt(
                current_program=parent.code,
                parent_program=parent.code,  # We don't have the parent's code, use the same
                program_metrics=parent.metrics,
                previous_programs=[p.to_dict() for p in self.database.get_top_programs(3)],
                top_programs=[p.to_dict() for p in inspirations],
                language=self.language,
                evolution_round=i,
                allow_full_rewrite=self.allow_full_rewrites,
                program_artifacts=parent_artifacts if parent_artifacts else None,
            )

            # Generate code modification
            try:
                llm_response = await self.llm_ensemble.generate_with_context(
                    system_message=prompt["system"],
                    messages=[{"role": "user", "content": prompt["user"]}],
                )

                # Parse the response
                if self.diff_based_evolution:
                    diff_blocks = extract_diffs(llm_response)

                    if not diff_blocks:
                        logger.warning(f"Iteration {i+1}: No valid diffs found in response")
                        continue

                    # Apply the diffs
                    child_code = apply_diff(parent.code, llm_response)
                    changes_summary = format_diff_summary(diff_blocks)
                else:
                    # Parse full rewrite
                    new_code = parse_full_rewrite(llm_response, self.language)

                    if not new_code:
                        logger.warning(f"Iteration {i+1}: No valid code found in response")
                        continue

                    child_code = new_code
                    changes_summary = "Full rewrite"

                # Check code length
                if len(child_code) > self.max_code_length:
                    logger.warning(
                        f"Iteration {i+1}: Generated code exceeds maximum length "
                        f"({len(child_code)} > {self.max_code_length})"
                    )
                    continue

                # Evaluate the child program
                child_id = str(uuid.uuid4())
                child_metrics = await self.evaluator.evaluate_program(child_code, child_id)

                # Handle artifacts if they exist
                artifacts = self.evaluator.get_pending_artifacts(child_id)

                # Create a child program
                child_program = Program(
                    id=child_id,
                    code=child_code,
                    language=self.language,
                    parent_id=parent.id,
                    generation=parent.generation + 1,
                    metrics=child_metrics,
                    metadata={
                        "changes": changes_summary,
                        "parent_metrics": parent.metrics,
                    },
                )

                # Add to database (will be added to current island)
                self.database.add(child_program, iteration=i + 1)

                # Store artifacts if they exist
                if artifacts:
                    self.database.store_artifacts(child_id, artifacts)

                # Increment generation for current island
                self.database.increment_island_generation()

                # Check if migration should occur
                if self.database.should_migrate():
                    logger.info(f"Performing migration at iteration {i+1}")
                    self.database.migrate_programs()
                    self.database.log_island_status()

                # Log progress
                iteration_time = time.time() - iteration_start
                self._log_iteration(i, parent, child_program, iteration_time)

                # Specifically check if this is the new best program
                if self.database.best_program_id == child_program.id:
                    logger.info(
                        f"🌟 New best solution found at iteration {i+1}: {child_program.id}"
                    )
                    logger.info(f"Metrics: {format_metrics_safe(child_program.metrics)}")

                # Save checkpoint
                if (i + 1) % self.checkpoint_interval == 0:
                    self._save_checkpoint(i + 1)
                    # Also log island status at checkpoints
                    logger.info(f"Island status at checkpoint {i+1}:")
                    self.database.log_island_status()

                # Check if target score reached
                if self.target_score is not None:
                    avg_score = sum(child_metrics.values()) / max(1, len(child_metrics))
                    if avg_score >= self.target_score:
                        logger.info(f"Target score {self.target_score} reached after {i+1} iterations")
                        break

            except Exception as e:
                logger.error(f"Error in iteration {i+1}: {str(e)}")
                continue

        # Get the best program using our tracking mechanism
        best_program = None
        if self.database.best_program_id:
            best_program = self.database.get(self.database.best_program_id)
            logger.info(f"Using tracked best program: {self.database.best_program_id}")

        # Fallback to calculating best program if tracked program not found
        if best_program is None:
            best_program = self.database.get_best_program()
            logger.info("Using calculated best program (tracked program not found)")

        # Check if there's a better program by combined_score that wasn't tracked
        if "combined_score" in best_program.metrics:
            best_by_combined = self.database.get_best_program(metric="combined_score")
            if (
                best_by_combined
                and best_by_combined.id != best_program.id
                and "combined_score" in best_by_combined.metrics
            ):
                # If the combined_score of this program is significantly better, use it instead
                if (
                    best_by_combined.metrics["combined_score"]
                    > best_program.metrics["combined_score"] + 0.02
                ):
                    logger.warning(
                        f"Found program with better combined_score: {best_by_combined.id}"
                    )
                    logger.warning(
                        f"Score difference: {best_program.metrics['combined_score']:.4f} vs {best_by_combined.metrics['combined_score']:.4f}"
                    )
                    best_program = best_by_combined

        if best_program:
            logger.info(
                f"Evolution complete. Best program has metrics: "
                f"{format_metrics_safe(best_program.metrics)}"
            )

            # Save the best program (using our tracked best program)
            self._save_best_program()

            return best_program
        else:
            logger.warning("No valid programs found during evolution")
            # Return None if no programs found instead of undefined initial_program
            return None

    def _log_iteration(
        self,
        iteration: int,
        parent: Program,
        child: Program,
        elapsed_time: float,
    ) -> None:
        """
        Log iteration progress

        Args:
            iteration: Iteration number
            parent: Parent program
            child: Child program
            elapsed_time: Elapsed time in seconds
        """
        # Calculate improvement using safe formatting
        improvement_str = format_improvement_safe(parent.metrics, child.metrics)

        logger.info(
            f"Iteration {iteration+1}: Child {child.id} from parent {parent.id} "
            f"in {elapsed_time:.2f}s. Metrics: "
            f"{format_metrics_safe(child.metrics)} "
            f"(Δ: {improvement_str})"
        )

    def _save_checkpoint(self, iteration: int) -> None:
        """
        Save a checkpoint

        Args:
            iteration: Current iteration number
        """
        checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Create specific checkpoint directory
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{iteration}")
        os.makedirs(checkpoint_path, exist_ok=True)

        # Save the database
        self.database.save(checkpoint_path, iteration)

        # Save the best program found so far
        best_program = None
        if self.database.best_program_id:
            best_program = self.database.get(self.database.best_program_id)
        else:
            best_program = self.database.get_best_program()

        if best_program:
            # Save the best program at this checkpoint
            best_program_path = os.path.join(checkpoint_path, f"best_program{self.file_extension}")
            with open(best_program_path, "w") as f:
                f.write(best_program.code)

            # Save metrics
            best_program_info_path = os.path.join(checkpoint_path, "best_program_info.json")
            with open(best_program_info_path, "w") as f:
                import json

                json.dump(
                    {
                        "id": best_program.id,
                        "generation": best_program.generation,
                        "iteration": best_program.iteration_found,
                        "current_iteration": iteration,
                        "metrics": best_program.metrics,
                        "language": best_program.language,
                        "timestamp": best_program.timestamp,
                        "saved_at": time.time(),
                    },
                    f,
                    indent=2,
                )

            logger.info(
                f"Saved best program at checkpoint {iteration} with metrics: "
                f"{format_metrics_safe(best_program.metrics)}"
            )

        logger.info(f"Saved checkpoint at iteration {iteration} to {checkpoint_path}")

    def _save_best_program(self, program: Optional[Program] = None) -> None:
        """
        Save the best program

        Args:
            program: Best program (if None, uses the tracked best program)
        """
        # If no program is provided, use the tracked best program from the database
        if program is None:
            if self.database.best_program_id:
                program = self.database.get(self.database.best_program_id)
            else:
                # Fallback to calculating best program if no tracked best program
                program = self.database.get_best_program()

        if not program:
            logger.warning("No best program found to save")
            return

        best_dir = os.path.join(self.output_dir, "best")
        os.makedirs(best_dir, exist_ok=True)

        # Use the extension from the initial program file
        filename = f"best_program{self.file_extension}"
        code_path = os.path.join(best_dir, filename)

        with open(code_path, "w") as f:
            f.write(program.code)

        # Save complete program info including metrics
        info_path = os.path.join(best_dir, "best_program_info.json")
        with open(info_path, "w") as f:
            import json

            json.dump(
                {
                    "id": program.id,
                    "generation": program.generation,
                    "iteration": program.iteration_found,
                    "timestamp": program.timestamp,
                    "parent_id": program.parent_id,
                    "metrics": program.metrics,
                    "language": program.language,
                    "saved_at": time.time(),
                },
                f,
                indent=2,
            )

        logger.info(f"Saved best program to {code_path} with program info to {info_path}")
