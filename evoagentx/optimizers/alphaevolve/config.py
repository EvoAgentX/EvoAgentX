# Acknowledgement: Modified from OpenEvolve (https://github.com/codelion/openevolve/blob/main/openevolve/config.py) under Apache-2.0 license

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from evoagentx.prompts.optimizers.alphaevolve_optimizer import (
    BASE_EVALUATOR_SYSTEM_TEMPLATE,
    BASE_SYSTEM_TEMPLATE,
)


@dataclass
class AlphaEvolvePromptConfig:
    """Configuration for prompt generation"""

    template_dir: Optional[str] = None
    system_message: str = BASE_SYSTEM_TEMPLATE
    evaluator_system_message: str = BASE_EVALUATOR_SYSTEM_TEMPLATE

    # Number of examples to include in the prompt
    num_top_programs: int = 3
    num_diverse_programs: int = 2

    # Template stochasticity
    use_template_stochasticity: bool = True
    template_variations: Dict[str, List[str]] = field(default_factory=dict)

    # Meta-prompting
    use_meta_prompting: bool = False
    meta_prompt_weight: float = 0.1

    # Artifact rendering
    include_artifacts: bool = True
    max_artifact_bytes: int = 20 * 1024  # 20KB in prompt
    artifact_security_filter: bool = True


@dataclass
class AlphaEvolveDatabaseConfig:
    """Configuration for the program database"""

    # General settings
    db_path: Optional[str] = None  # Path to store database on disk
    in_memory: bool = True

    # Evolutionary parameters
    population_size: int = 1000
    archive_size: int = 100
    num_islands: int = 5

    # Selection parameters
    elite_selection_ratio: float = 0.1
    exploration_ratio: float = 0.2
    exploitation_ratio: float = 0.7
    diversity_metric: str = "edit_distance"  # Options: "edit_distance", "feature_based"

    # Feature map dimensions for MAP-Elites
    feature_dimensions: List[str] = field(default_factory=lambda: ["score", "complexity"])
    feature_bins: int = 10

    # Migration parameters for island-based evolution
    migration_interval: int = 50  # Migrate every N generations
    migration_rate: float = 0.1  # Fraction of population to migrate

    # Random seed for reproducible sampling
    random_seed: Optional[int] = None

    # Artifact storage
    artifacts_base_path: Optional[str] = None  # Defaults to db_path/artifacts
    artifact_size_threshold: int = 32 * 1024  # 32KB threshold
    cleanup_old_artifacts: bool = True
    artifact_retention_days: int = 30

