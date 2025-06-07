import json
import logging
import os
import pkgutil
import random
import re
import sys
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import Field
from tqdm.asyncio import tqdm_asyncio

import wandb
from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    EvalHandlingEnum,
    Item,
    ScoredDataGroup,
)
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

# Add the local reasoning-gym submodule to Python's path
# This allows `import reasoning_gym` to find the local submodule
_SUBMODULE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "reasoning-gym")
)
if _SUBMODULE_DIR not in sys.path:
    sys.path.insert(0, _SUBMODULE_DIR)

# Attempt to import reasoning_gym. If not found, a warning will be issued in _get_task_names.
try:
    import reasoning_gym
    from reasoning_gym.utils import extract_answer
except ImportError:
    reasoning_gym = None
    extract_answer = None


system_prompt = (
    "You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the "
    "problem and deliberate with yourself via systematic reasoning processes to help come to a correct "
    "solution prior to answering. You should enclose your thoughts and internal monologue inside <think> "
    "</think> tags, and then provide your solution or response to the problem. After your thinking, "
    "make sure to clearly provide your final answer inside <answer></answer> tags. You can provide "
    "context, explanation, etc before and after your answer tags/answer, but you must provide a single "
    "answer and place it inside <answer> tags. You must provide a single answer and place it inside "
    "<answer> tags."
)


class ReasoningGymEnvConfig(BaseEnvConfig):
    """Extended configuration for ReasoningGymEnv with additional fields."""

    dump_rollouts: bool = Field(
        default=False,
        description="Whether to dump successful rollouts (above threshold) to JSONL files.",
    )
    dump_failed_rollouts: bool = Field(
        default=False,
        description="Whether to dump failed rollouts (all 0 scores) to JSONL files for debugging.",
    )
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility.",
    )
    debug_logging: bool = Field(
        default=False,
        description="Enable debug-level logging for more verbose output.",
    )
    suppress_base_env_logs: bool = Field(
        default=True,
        description="Suppress verbose base environment logs (like status dict updates).",
    )
    rollout_save_score_threshold: float = Field(
        default=0.7,
        description="Minimum score threshold for saving rollouts to data dumps. Only groups with at least one rollout above this threshold will be saved.",  # noqa: E501
    )
    num_eval_samples_per_task: int = Field(
        default=5,
        description="Number of evaluation samples to generate per task for the test set.",
    )
    eval_seed: int = Field(
        default=123,
        description="Seed for generating fixed evaluation set to ensure reproducibility.",
    )

    def validate_config(self):
        """Validate configuration parameters."""
        if not (0.0 <= self.rollout_save_score_threshold <= 1.0):
            raise ValueError(
                f"rollout_save_score_threshold must be between 0.0 and 1.0, got {self.rollout_save_score_threshold}"
            )
        if self.rollout_save_score_threshold == 1.0:
            print(
                f"Warning: rollout_save_score_threshold is {self.rollout_save_score_threshold}, which may be too strict and result in no saved rollouts."  # noqa: E501
            )
        if self.num_eval_samples_per_task <= 0:
            raise ValueError(
                f"num_eval_samples_per_task must be positive, got {self.num_eval_samples_per_task}"
            )


class ReasoningGymEnv(BaseEnv):
    name = "reasoning_gym"
    env_config_cls = ReasoningGymEnvConfig

    def __init__(
        self,
        config: ReasoningGymEnvConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        # Validate configuration before proceeding
        config.validate_config()
        super().__init__(config, server_configs, slurm, testing)

        # Initialize the logger like swe_rl_env.py
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            # Add a basic stream handler if no handlers are configured
            _handler = logging.StreamHandler()
            _formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            _handler.setFormatter(_formatter)
            self.logger.addHandler(_handler)
            # Set logging level based on config
            log_level = logging.DEBUG if self.config.debug_logging else logging.INFO
            self.logger.setLevel(log_level)
        # Ensure the logger itself is enabled
        self.logger.disabled = False

        # Suppress base environment logs if requested
        if self.config.suppress_base_env_logs:
            # Set the base environment logger to WARNING level to suppress INFO logs
            base_logger = logging.getLogger("atroposlib.envs.base")
            base_logger.setLevel(logging.WARNING)

        # Set max_token_len for base class compatibility
        self.max_token_len = self.config.max_token_length

        self.percent_correct_buffer = list()
        self.eval_metrics = list()
        self.rollouts_for_wandb: List[List[Tuple[str, float, str, str]]] = []
        self.task_names: List[str] = []
        self.test_items_with_scorers: List[Tuple[Dict[str, Any], Any]] = []
        self.rng = random.Random()

        self.run_uuid = str(uuid.uuid4())
        self.rollouts_to_save_buffer: List[
            Dict[str, Union[str, List[Dict[str, Union[List[Dict[str, str]], float]]]]]
        ] = []
        self.processed_item_count = 0
        self.datadumps_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data_dumps"
        )
        self.save_file_batch_num = 0

        # For saving failed rollouts (all 0 scores) for debugging
        self.failed_rollouts_to_save_buffer: List[
            Dict[str, Union[str, List[Dict[str, Union[List[Dict[str, str]], float]]]]]
        ] = []
        self.failed_processed_item_count = 0
        self.failed_save_file_batch_num = 0

    @classmethod
    def config_init(cls) -> Tuple[ReasoningGymEnvConfig, List[APIServerConfig]]:
        env_config = ReasoningGymEnvConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
            group_size=16,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=250,
            seed=1918,
            batch_size=1024,
            steps_per_eval=25,
            max_token_length=1024 * 16,
            inference_weight=1.0,
            wandb_name="reasoning_gym_think",  # Specific name for reasoning gym
            eval_handling=EvalHandlingEnum.LIMIT_TRAIN,
            eval_limit_ratio=0.1,
            num_rollouts_per_group_for_logging=4,
            num_rollouts_to_keep=50,
            dump_rollouts=False,
            dump_failed_rollouts=False,
            debug_logging=False,
            suppress_base_env_logs=True,
            rollout_save_score_threshold=0.51,
            num_eval_samples_per_task=5,
            eval_seed=123,
        )
        server_configs = [
            APIServerConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
                base_url="http://localhost:9004/v1",
                api_key="x",
                num_max_requests_at_once=32,
                num_requests_for_eval=256,
            ),
        ]
        return env_config, server_configs

    def _get_task_names(self) -> List[str]:
        """
        Gets task names from the reasoning_gym DATASETS registry.
        This is more reliable than dynamic discovery via pkgutil.
        Falls back to a predefined list if registry access fails.
        """
        if reasoning_gym is None:
            print(
                "ERROR: The local reasoning-gym submodule could not be imported. "
                "Ensure it is present in 'atropos/environments/reasoning_gym_environment/reasoning-gym' "
                "and is a valid Python package (e.g., has __init__.py)."
            )
            return ["algebra/linear_1d", "arithmetic/add_or_subtract", "leg_counting"]

        discovered_tasks = []
        try:
            # Access the DATASETS registry directly from reasoning_gym.factory
            from reasoning_gym.factory import DATASETS

            discovered_tasks = list(DATASETS.keys())
            print(
                f"Discovered {len(discovered_tasks)} tasks from DATASETS registry: {discovered_tasks[:10]}{'...' if len(discovered_tasks) > 10 else ''}"  # noqa: E501
            )

        except Exception as e:
            print(
                f"WARNING: Could not access DATASETS registry: {e}. "
                "Falling back to manual discovery and validation."
            )

            # Fallback to pkgutil discovery if registry access fails
            try:
                package = reasoning_gym
                if not hasattr(package, "__path__"):
                    raise AttributeError("'reasoning_gym' is not a package")

                base_module_path_parts = package.__name__.split(".")

                print(
                    f"Attempting to discover tasks in package: {package.__name__} from path: {package.__path__}"
                )

                for _, modname, ispkg in pkgutil.walk_packages(
                    path=package.__path__,
                    prefix=package.__name__ + ".",
                    onerror=lambda name: print(
                        f"Error importing module during task discovery: {name}"
                    ),
                ):
                    if not ispkg:
                        module_parts = modname.split(".")
                        if (
                            len(module_parts) > len(base_module_path_parts)
                            and module_parts[: len(base_module_path_parts)]
                            == base_module_path_parts
                        ):

                            task_specific_parts = module_parts[
                                len(base_module_path_parts) :
                            ]

                            # Filter out potential private/internal modules
                            if any(
                                part.startswith("_") for part in task_specific_parts
                            ):
                                continue

                            task_name = "/".join(task_specific_parts)
                            discovered_tasks.append(task_name)

                if discovered_tasks:
                    print(
                        f"Dynamically discovered {len(discovered_tasks)} potential task names from submodule."
                    )

            except Exception as e2:
                print(
                    f"WARNING: Pkgutil discovery also failed: {e2}. Using fallback list."
                )

        if not discovered_tasks:
            print(
                "WARNING: All discovery methods failed. Using fallback list of known tasks."
            )
            # Complete fallback list with all available reasoning-gym tasks
            fallback_tasks = [
                "ab",
                "acre",
                "advanced_geometry",
                "aiw",
                "arc_1d",
                "arc_agi",
                "base_conversion",
                "basic_arithmetic",
                "bf",
                "binary_alternation",
                "binary_matrix",
                "bitwise_arithmetic",
                "boxnet",
                "caesar_cipher",
                "calendar_arithmetic",
                "chain_sum",
                "circuit_logic",
                "codeio",
                "color_cube_rotation",
                "complex_arithmetic",
                "composite",
                "count_bits",
                "count_primes",
                "countdown",
                "course_schedule",
                "cryptarithm",
                "decimal_arithmetic",
                "decimal_chain_sum",
                "dice",
                "emoji_mystery",
                "family_relationships",
                "figlet_font",
                "fraction_simplification",
                "futoshiki",
                "game_of_life",
                "game_of_life_halting",
                "gcd",
                "graph_color",
                "group_anagrams",
                "gsm_symbolic",
                "intermediate_integration",
                "isomorphic_strings",
                "jugs",
                "knight_swap",
                "knights_knaves",
                "largest_island",
                "lcm",
                "leg_counting",
                "letter_counting",
                "letter_jumble",
                "list_functions",
                "mahjong_puzzle",
                "manipulate_matrix",
                "maze",
                "mini_sudoku",
                "modulo_grid",
                "n_queens",
                "needle_haystack",
                "number_filtering",
                "number_format",
                "number_sequence",
                "number_sorting",
                "palindrome_generation",
                "palindrome_partitioning",
                "polynomial_equations",
                "polynomial_multiplication",
                "pool_matrix",
                "power_function",
                "prime_factorization",
                "products",
                "propositional_logic",
                "puzzle24",
                "quantum_lock",
                "ransom_note",
                "rearc",
                "rectangle_count",
                "rotate_matrix",
                "rotten_oranges",
                "rubiks_cube",
                "rush_hour",
                "self_reference",
                "sentence_reordering",
                "shortest_path",
                "simple_equations",
                "simple_geometry",
                "simple_integration",
                "sokoban",
                "spell_backward",
                "spiral_matrix",
                "string_insertion",
                "string_manipulation",
                "string_splitting",
                "string_synthesis",
                "sudoku",
                "syllogism",
                "time_intervals",
                "tower_of_hanoi",
                "tsumego",
                "word_ladder",
                "word_sequence_reversal",
                "word_sorting",
                "zebra_puzzles",
            ]
            return self._validate_discovered_tasks(
                fallback_tasks, reasoning_gym if reasoning_gym else None
            )

        # Validate all discovered tasks
        return self._validate_discovered_tasks(discovered_tasks, reasoning_gym)

    def _validate_discovered_tasks(
        self, task_names_to_validate: List[str], rg_package_or_none: Any
    ) -> List[str]:
        """
        Validates a list of task names by attempting to create a dataset for each.
        Args:
            task_names_to_validate: List of task names (e.g., "domain/task").
            rg_package_or_none: The imported reasoning_gym package, or None if import failed.
        Returns:
            A list of task names that were successfully validated.
        """
        if rg_package_or_none is None:
            print(
                "Validation SKIPPED: reasoning_gym package not available for validation."
            )
            return []

        valid_tasks = []
        print(
            f"Validating {len(task_names_to_validate)} discovered/fallback task names..."
        )
        for task_name in task_names_to_validate:
            try:
                # Use the potentially imported reasoning_gym directly for create_dataset
                _ = rg_package_or_none.create_dataset(task_name, size=1, seed=0)
                valid_tasks.append(task_name)
            except Exception as e:
                print(
                    f"Note: Task '{task_name}' could not be loaded (validation failed): {type(e).__name__} - {e}"
                )  # noqa: E501
                pass

        if not valid_tasks and task_names_to_validate:
            print(
                "WARNING: Validation of discovered/fallback tasks failed for all. This might indicate a systematic issue."  # noqa: E501
            )
            # If validation fails for all, it's safer to return an empty list or a minimal known-good set.
            # However, if task_names_to_validate was non-empty, returning it raw might be a last resort if validation itself is flawed. # noqa: E501
            # For safety, let's prefer an empty list if validation fails completely.
            print("No tasks passed validation.")
            return []

        if not valid_tasks:
            print(
                "CRITICAL WARNING: No valid reasoning-gym tasks could be loaded. Environment may not function."
            )
            # Return an absolute minimal, known-good task if everything else fails.
            # This specific task must exist and be loadable in reasoning_gym.
            try:
                _ = rg_package_or_none.create_dataset("leg_counting", size=1, seed=0)
                print("Falling back to absolute minimal task: 'leg_counting'")
                return ["leg_counting"]
            except Exception:
                print("Absolute fallback 'leg_counting' also failed to load.")
                return []

        print(f"Validated {len(valid_tasks)} tasks for use.")
        return valid_tasks

    async def setup(self):
        # The reasoning_gym import is now handled at the top with sys.path modification.
        if reasoning_gym is None:
            raise ImportError(
                "reasoning-gym library could not be imported from the local submodule. "
                "This environment cannot function. Check submodule presence and integrity."
            )

        self.logger.info("Setting up ReasoningGym environment...")

        self.task_names = (
            self._get_task_names()
        )  # _get_task_names now uses self._validate_discovered_tasks
        if not self.task_names:
            raise ValueError(
                "No reasoning_gym tasks could be loaded. Environment setup failed."
            )

        self.logger.info(
            f"ReasoningGymEnv: Initialized with {len(self.task_names)} tasks."
        )
        self.logger.info(f"Sample tasks: {self.task_names[:5]}")

        # Seed for main RNG used in get_next_item
        # The seed for reasoning_gym dataset creation will be self.iter
        self.rng.seed(self.config.seed)
        self.iter = 0

        # Create a fixed test set for evaluation
        self.logger.info("Generating fixed test set for evaluation...")
        eval_tasks_sample = self.rng.sample(
            self.task_names, min(len(self.task_names), 20)
        )  # Sample 20 tasks for eval

        for task_name in tqdm_asyncio(eval_tasks_sample, desc="Creating eval dataset"):
            try:
                # Each task gets its own dataset instance for evaluation
                # Using a fixed seed for reproducibility of the test set
                dataset = reasoning_gym.create_dataset(
                    task_name,
                    size=self.config.num_eval_samples_per_task,
                    seed=self.config.eval_seed,
                )
                for item in dataset:
                    self.test_items_with_scorers.append((item, dataset))
            except Exception as e:
                self.logger.warning(
                    f"Could not create eval dataset for task '{task_name}': {e}"
                )

        if not self.test_items_with_scorers:
            self.logger.warning(
                "No evaluation items could be generated. Evaluation might be skipped or fail."
            )
        else:
            self.logger.info(
                f"Generated {len(self.test_items_with_scorers)} items for the evaluation test set."
            )

        self.logger.info(
            "ReasoningGym environment setup complete. Ready to start training!"
        )
        self.logger.info(
            f"Configuration: group_size={self.config.group_size}, max_token_length={self.config.max_token_length}, steps_per_eval={self.config.steps_per_eval}"  # noqa: E501
        )
        if self.config.dump_rollouts:
            self.logger.info(
                f"Data dumping enabled with score threshold: {self.config.rollout_save_score_threshold}"
            )
        self.logger.info(
            "Using strict <answer> tag enforcement: models must use exactly one <answer> tag or receive 0 score"
        )
        self.logger.info(
            "Using dual-format scoring for valid answers: trying both raw answers and <answer>-tagged answers, using higher score"  # noqa: E501
        )

    async def get_next_item(self) -> Optional[Item]:
        """
        Get the next training item by randomly selecting a reasoning_gym task and generating a sample.
        Returns:
            A tuple: (prompt_messages, reasoning_gym_item, reasoning_gym_dataset_object)
            prompt_messages: Formatted for the language model.
            reasoning_gym_item: The raw item from reasoning_gym (dict with 'question', 'answer', 'metadata').
            reasoning_gym_dataset_object: The dataset object from which the item was generated (for scoring).
        """
        if not self.task_names:
            return None  # Should not happen if setup is correct

        selected_task_name = self.rng.choice(self.task_names)

        try:
            # Create a new dataset instance for each item to ensure variety if tasks have internal state
            # Use self.iter for seed to get different questions over time
            current_seed = self.config.seed + self.iter  # Vary seed per item
            dataset_obj = reasoning_gym.create_dataset(
                selected_task_name, size=1, seed=current_seed
            )
            rg_item = next(iter(dataset_obj))  # Get the single item

            # Log task selection every 10 items to avoid spam
            if self.iter % 10 == 0:
                self.logger.info(
                    f"Selected task: {selected_task_name} (iteration {self.iter})"
                )

        except Exception as e:
            self.logger.warning(
                f"Error generating item for task {selected_task_name} with seed {current_seed}: {e}"
            )
            return None  # Skip this item if generation fails

        self.iter += 1

        question_text = rg_item["question"]

        # Construct prompt messages
        prompt_messages = [
            frozenset({"role": "system", "content": system_prompt}.items()),
            frozenset({"role": "user", "content": question_text}.items()),
        ]

        # The 'answer' here is the data needed for scoring later
        return (tuple(prompt_messages), rg_item, dataset_obj)

    def _extract_final_answer(self, text: str) -> Optional[str]:
        """
        Extracts the content from <answer> tags using reasoning-gym's extract_answer function.
        Enforces strict compliance with answer tag instructions.
        Returns None if model doesn't follow instructions properly (no answer tags or multiple answer tags outside think).
        """  # noqa: E501
        if extract_answer is None:
            # If reasoning-gym not available, we can't enforce the format
            if self.config.debug_logging:
                self.logger.debug(
                    "reasoning-gym extract_answer not available, cannot enforce answer tag format"
                )
            return None

        # Check for multiple <answer> tags outside of <think> blocks
        # First, remove all <think>...</think> content to check only the "outside" content
        think_pattern = r"<think>.*?</think>"
        text_outside_think = re.sub(
            think_pattern, "", text, flags=re.DOTALL | re.IGNORECASE
        )

        # Count <answer> tags in the content outside <think> blocks
        answer_pattern = r"<answer>.*?</answer>"
        answer_matches_outside_think = re.findall(
            answer_pattern, text_outside_think, flags=re.DOTALL | re.IGNORECASE
        )

        if len(answer_matches_outside_think) > 1:
            if self.config.debug_logging:
                self.logger.debug(
                    f"Model provided {len(answer_matches_outside_think)} answer tags outside think blocks - failing for not following single answer instruction"  # noqa: E501
                )
            return None

        # Try to extract from <answer> tags using reasoning-gym's function
        answer_content = extract_answer(text, tag_name="answer", strip=True)
        if answer_content is not None:
            if self.config.debug_logging:
                self.logger.debug(
                    f"Successfully extracted answer from <answer> tags: '{answer_content[:100]}{'...' if len(answer_content) > 100 else ''}'"  # noqa: E501
                )
            return answer_content

        # No valid answer tags found - model failed to follow instructions
        if self.config.debug_logging:
            self.logger.debug(
                "No <answer> tags found - model failed to follow answer format instructions"
            )
        return None

    def _score_answer_with_both_formats(
        self, model_answer: str, rg_item: dict, dataset_obj: Any
    ) -> float:
        """
        Score the answer using both formats (with and without <answer> tags) and return the higher score.
        This handles verifiers that expect different formats.
        """
        # Format 1: Just the answer content
        try:
            score1 = dataset_obj.score_answer(answer=model_answer, entry=rg_item)
            score1 = max(0.0, min(1.0, float(score1)))
        except Exception as e:
            task_name = rg_item.get("metadata", {}).get("source_dataset", "unknown")
            self.logger.debug(
                f"Error scoring answer format 1 for task {task_name}: {e}"
            )
            score1 = 0.0

        # Format 2: Answer wrapped in <answer> tags
        answer_with_tags = f"<answer>{model_answer}</answer>"
        try:
            score2 = dataset_obj.score_answer(answer=answer_with_tags, entry=rg_item)
            score2 = max(0.0, min(1.0, float(score2)))
        except Exception as e:
            task_name = rg_item.get("metadata", {}).get("source_dataset", "unknown")
            self.logger.debug(
                f"Error scoring answer format 2 for task {task_name}: {e}"
            )
            score2 = 0.0

        # Return the higher score
        final_score = max(score1, score2)

        # Log which format worked better (only in debug mode)
        if self.config.debug_logging and score1 != score2:
            task_name = rg_item.get("metadata", {}).get("source_dataset", "unknown")
            if score1 > score2:
                self.logger.debug(
                    f"Task {task_name}: Raw answer format scored higher ({score1:.3f} vs {score2:.3f})"
                )
            else:
                self.logger.debug(
                    f"Task {task_name}: Tagged answer format scored higher ({score2:.3f} vs {score1:.3f})"
                )

        return final_score

    async def score(
        self,
        rollout_group_data: List[Tuple[Tuple[Dict[str, str]], Dict[str, Any], Any]],
    ) -> Optional[ScoredDataGroup]:
        """
        Scores a group of rollouts using reasoning_gym's score_answer method.
        Args:
            rollout_group_data: A list of tuples, where each tuple contains:
                - trajectory_messages: The full conversation history for the rollout.
                - rg_item: The original reasoning_gym item (contains 'question', 'answer', 'metadata').
                - dataset_obj: The reasoning_gym dataset object used to generate and score the item.
        Returns:
            ScoredDataGroup with scores between 0.0 and 1.0, or None if no valid items
        """
        scores_container = ScoredDataGroup()
        scores_container["tokens"] = list()
        scores_container["masks"] = list()
        scores_container["scores"] = list()

        if not rollout_group_data:
            return None

        rg_item_for_group = rollout_group_data[0][1]
        dataset_obj_for_group = rollout_group_data[0][2]

        # Shuffle to avoid bias in selection
        random.shuffle(rollout_group_data)

        for trajectory_messages, _, _ in rollout_group_data:
            model_full_response = trajectory_messages[-1]["content"]

            # Extract the part of the response that should be the answer
            model_answer_to_score = self._extract_final_answer(model_full_response)

            # If extraction failed (model didn't follow instructions), give 0 score
            if model_answer_to_score is None:
                reward_0_to_1 = 0.0
                if self.config.debug_logging:
                    task_name = rg_item_for_group.get("metadata", {}).get(
                        "source_dataset", "unknown"
                    )
                    self.logger.debug(
                        f"Task {task_name}: Giving 0 score due to failed answer extraction (didn't follow format)"
                    )
            else:
                # Use our dual-format scoring method that tries both raw answer and tagged answer
                reward_0_to_1 = self._score_answer_with_both_formats(
                    model_answer_to_score, rg_item_for_group, dataset_obj_for_group
                )

            # Tokenize the conversation for learning
            out_dict = tokenize_for_trainer(self.tokenizer, trajectory_messages)
            tokens = out_dict["tokens"]
            masks = out_dict["masks"]

            # Remove examples with insufficient context
            if len([1 for i in masks if i != -100]) < 10:
                continue

            scores_container["tokens"].append(tokens)
            scores_container["masks"].append(masks)
            scores_container["scores"].append(reward_0_to_1)

            if len(scores_container["tokens"]) >= self.config.group_size:
                break

        if not scores_container["tokens"]:
            self.logger.warning(
                "No valid items were scored in this group - all items had insufficient context or failed scoring"
            )
            return None

        # Record success rate metrics (convert to binary for percent_correct tracking)
        for score_val in scores_container["scores"]:
            self.percent_correct_buffer.append(1.0 if score_val >= 0.5 else 0.0)

        # Calculate and log average score for the current group
        current_scores = scores_container.get("scores", [])
        if current_scores:
            average_score = sum(current_scores) / len(current_scores)
            task_name = rg_item_for_group.get("metadata", {}).get(
                "source_dataset", "unknown_task"
            )
            log_message_main = (
                f"Task: {task_name} | Group average score: {average_score:.4f}"
            )
            if all(s >= 0.5 for s in current_scores):
                self.logger.info(f"{log_message_main} (All correct in this group!)")
            elif all(s == 0.0 for s in current_scores):
                self.logger.info(f"{log_message_main} (All failed - no valid answers!)")
            elif all(s < 0.5 for s in current_scores):
                self.logger.info(
                    f"{log_message_main} (All incorrect but some partial credit!)"
                )
            else:
                self.logger.info(log_message_main)

        # Apply length penalty if all responses are correct (score >= 0.5)
        if all(s >= 0.5 for s in scores_container["scores"]):
            # Calculate token lengths
            token_lengths = [len(token) for token in scores_container["tokens"]]
            if max(token_lengths) == 0:
                return None

            # Get max allowed token length from config
            max_allowed_length = self.config.max_token_length
            # Set threshold at 75% of max_token_length
            length_threshold = max_allowed_length * 0.75

            # Apply modified length penalty with threshold
            new_scores = []
            penalties_applied = 0
            for i, length in enumerate(token_lengths):
                original_score = scores_container["scores"][i]
                if length <= length_threshold:
                    new_scores.append(original_score)
                else:
                    # Calculate how far we are between threshold and max as a percentage
                    percentage_of_range = (length - length_threshold) / (
                        max_allowed_length - length_threshold
                    )
                    percentage_of_range = min(percentage_of_range, 1.0)
                    # Apply linear penalty scaling from original_score down to 0.0
                    penalized_score = original_score * (1.0 - percentage_of_range)
                    new_scores.append(penalized_score)
                    penalties_applied += 1

            if penalties_applied > 0:
                avg_length = sum(token_lengths) / len(token_lengths)
                self.logger.debug(
                    f"Applied length penalty to {penalties_applied}/{len(token_lengths)} responses (avg length: {avg_length:.0f}, threshold: {length_threshold:.0f})"  # noqa: E501
                )

            scores_container["scores"] = new_scores

        # Check if all scores are the same (no learning signal)
        if all(
            scores_container["scores"][0] == score
            for score in scores_container["scores"]
        ):
            self.logger.debug(
                f"All scores in group are identical ({scores_container['scores'][0]:.4f}) - no learning signal, skipping group"  # noqa: E501
            )

            # Before returning None, check if this is a completely failed group (all 0.0 scores) for debugging
            if self.config.dump_failed_rollouts and all(
                score == 0.0 for score in scores_container["scores"]
            ):
                self.logger.debug(
                    "Saving failed group (all 0 scores) for debugging analysis"
                )
                await self._save_failed_group_for_debugging(
                    rollout_group_data, scores_container
                )

            return None

        return scores_container

    async def _save_failed_group_for_debugging(
        self, rollout_group_data, scores_container
    ):
        """Helper method to save failed groups (all 0 scores) for debugging analysis."""
        failed_rollouts_with_scores_to_save = []

        # Build the failed rollouts data structure
        for i, (trajectory_messages, rg_item, dataset_obj) in enumerate(
            rollout_group_data
        ):
            if i < len(scores_container["scores"]):
                score_for_rollout = scores_container["scores"][i]
                failed_rollouts_with_scores_to_save.append(
                    {
                        "conversation": trajectory_messages,  # Full conversation history
                        "score": score_for_rollout,
                    }
                )

        if failed_rollouts_with_scores_to_save:
            # Extract item info for logging - get from first rollout
            _, rg_item, _ = rollout_group_data[0]
            item_id = rg_item.get("metadata", {}).get("source_dataset", "unknown_task")

            failed_item_data_to_save = {
                "item_id": item_id,
                "rollouts": failed_rollouts_with_scores_to_save,
            }
            self.failed_rollouts_to_save_buffer.append(failed_item_data_to_save)
            self.failed_processed_item_count += 1

            # Calculate progress toward next failed save
            failed_batch_progress = (
                self.failed_processed_item_count % 50
            )  # Save failed every 50 items
            if failed_batch_progress == 0:
                failed_batch_progress = (
                    50  # Show 50/50 instead of 0/50 when we hit the threshold
                )

            # Log progress every 10 failed items or when we hit the save threshold
            if failed_batch_progress % 10 == 0 or failed_batch_progress == 50:
                self.logger.info(
                    f"Failed rollouts progress: {failed_batch_progress}/50 items buffered "
                    f"(Total failed processed: {self.failed_processed_item_count}, Failed buffer size: {len(self.failed_rollouts_to_save_buffer)})"  # noqa: E501
                )

            # Check if it's time to save a batch of failed rollouts (every 50 instead of 100)
            if (
                self.config.dump_failed_rollouts
                and self.failed_processed_item_count % 50 == 0
                and self.failed_processed_item_count > 0
            ):
                failed_log_msg = (
                    f"Reached {self.failed_processed_item_count} failed items. "
                    f"Triggering save for {len(self.failed_rollouts_to_save_buffer)} failed items "
                    f"(each with multiple failed rollouts)."
                )
                self.logger.info(failed_log_msg)
                await self._save_failed_rollouts_to_jsonl()

    async def collect_trajectories(
        self, item: Item
    ) -> Tuple[Optional[ScoredDataGroup], List]:
        """
        Generate model responses for the given item and then score them.
        Args:
            item: A tuple from get_next_item: (prompt_messages, rg_item, dataset_obj)
        """
        prompt_messages_tuple, rg_item, dataset_obj = item

        # Apply chat template to convert messages to a single string
        # Ensure prompt_messages_tuple is correctly formatted list of dicts
        messages_for_template = [dict(msg_fset) for msg_fset in prompt_messages_tuple]

        prompt_str = self.tokenizer.apply_chat_template(
            messages_for_template, add_generation_prompt=True, tokenize=False
        )

        # Calculate max_tokens like tool_calling_server
        prompt_tokens = len(self.tokenizer.encode(prompt_str))
        max_tokens = min(1024 * 15, self.config.max_token_length - prompt_tokens)

        completions = await self.server.completion(
            prompt=prompt_str,
            n=self.config.group_size,
            max_tokens=max_tokens,
            temperature=0.8,
        )

        to_score_list = []
        for choice in completions.choices:
            self.completion_lengths.append(len(choice.text))

            # Create full trajectory messages for this choice
            current_trajectory_messages = list(messages_for_template)
            current_trajectory_messages.append(
                {"role": "assistant", "content": choice.text}
            )

            to_score_list.append(
                (tuple(current_trajectory_messages), rg_item, dataset_obj)
            )

        scored_data_group = await self.score(to_score_list)

        # If rollouts were generated and scored, and data dumping is enabled, prepare them for saving
        if scored_data_group and self.config.dump_rollouts:
            # Only save groups that have at least one rollout with score > threshold
            group_scores = scored_data_group.get("scores", [])
            threshold = self.config.rollout_save_score_threshold
            if any(score > threshold for score in group_scores):
                self.logger.debug(
                    f"Saving group with scores: {[f'{s:.3f}' for s in group_scores]} (has high-quality rollout, threshold: {threshold})"  # noqa: E501
                )
                rollouts_with_scores_to_save = []

                num_scored_rollouts = len(group_scores)
                for i in range(num_scored_rollouts):
                    conversation_messages = to_score_list[i][0]
                    score_for_rollout = group_scores[i]
                    rollouts_with_scores_to_save.append(
                        {
                            "conversation": conversation_messages,
                            "score": score_for_rollout,
                        }
                    )

                if rollouts_with_scores_to_save:
                    # Extract item info for logging
                    _, rg_item, _ = item
                    item_id = rg_item.get("metadata", {}).get(
                        "source_dataset", "unknown_task"
                    )

                    item_data_to_save = {
                        "item_id": item_id,
                        "rollouts": rollouts_with_scores_to_save,
                    }
                    self.rollouts_to_save_buffer.append(item_data_to_save)
                    self.processed_item_count += 1

                # Calculate progress toward next save
                current_batch_progress = self.processed_item_count % 100
                if current_batch_progress == 0:
                    current_batch_progress = 100

                # Log progress every 10 items or when we hit the save threshold
                if current_batch_progress % 10 == 0 or current_batch_progress == 100:
                    self.logger.info(
                        f"Data dump progress: {current_batch_progress}/100 items buffered "
                        f"(Total processed: {self.processed_item_count}, Buffer size: {len(self.rollouts_to_save_buffer)})"  # noqa: E501
                    )

                # Check if it's time to save a batch of rollouts
                if (
                    self.config.dump_rollouts
                    and self.processed_item_count % 100 == 0
                    and self.processed_item_count > 0
                ):
                    log_msg = (
                        f"Reached {self.processed_item_count} processed items. "
                        f"Triggering save for {len(self.rollouts_to_save_buffer)} items "
                        f"(each with multiple scored rollouts)."
                    )
                    self.logger.info(log_msg)
                    await self._save_rollouts_to_jsonl()
            else:
                max_score = max(group_scores) if group_scores else 0.0
                self.logger.debug(
                    f"Skipping group save - no high-quality rollouts (max score: {max_score:.3f}, threshold: {threshold})"  # noqa: E501
                )

        to_backlog = []
        return scored_data_group, to_backlog

    async def _save_rollouts_to_jsonl(self):
        """Saves the buffered rollouts to a JSONL file in the datadumps directory."""
        if not self.rollouts_to_save_buffer:
            self.logger.info("No rollouts in buffer to save.")
            return

        try:
            if not os.path.exists(self.datadumps_dir):
                os.makedirs(self.datadumps_dir)
                self.logger.info(f"Created directory: {self.datadumps_dir}")
        except OSError as e:
            self.logger.error(f"Error creating directory {self.datadumps_dir}: {e}")
            return

        file_path = os.path.join(
            self.datadumps_dir,
            f"reasoning_gym_environment_rollouts_{self.run_uuid}_{self.save_file_batch_num:04d}.jsonl",
        )

        try:
            with open(file_path, "w") as f:
                for rollout_dict in self.rollouts_to_save_buffer:
                    json.dump(rollout_dict, f)
                    f.write("\n")
            self.logger.info(
                f"Successfully saved {len(self.rollouts_to_save_buffer)} rollouts to {file_path}"
            )
            self.rollouts_to_save_buffer.clear()
            self.save_file_batch_num += 1
        except IOError as e:
            self.logger.error(f"Error writing rollouts to {file_path}: {e}")
        except Exception as e:
            self.logger.error(
                f"An unexpected error occurred while saving rollouts to {file_path}: {e}"
            )

    async def _save_failed_rollouts_to_jsonl(self):
        """Saves the buffered failed rollouts (all 0 scores) to a JSONL file for debugging."""
        if not self.failed_rollouts_to_save_buffer:
            self.logger.info("No failed rollouts in buffer to save.")
            return

        try:
            if not os.path.exists(self.datadumps_dir):
                os.makedirs(self.datadumps_dir)
                self.logger.info(f"Created directory: {self.datadumps_dir}")
        except OSError as e:
            self.logger.error(f"Error creating directory {self.datadumps_dir}: {e}")
            return

        file_path = os.path.join(
            self.datadumps_dir,
            f"reasoning_gym_environment_FAILED_rollouts_{self.run_uuid}_{self.failed_save_file_batch_num:04d}.jsonl",
        )

        try:
            with open(file_path, "w") as f:
                for rollout_dict in self.failed_rollouts_to_save_buffer:
                    json.dump(rollout_dict, f)
                    f.write("\n")
            self.logger.info(
                f"Successfully saved {len(self.failed_rollouts_to_save_buffer)} FAILED rollouts to {file_path}"
            )
            self.failed_rollouts_to_save_buffer.clear()
            self.failed_save_file_batch_num += 1
        except IOError as e:
            self.logger.error(f"Error writing failed rollouts to {file_path}: {e}")
        except Exception as e:
            self.logger.error(
                f"An unexpected error occurred while saving failed rollouts to {file_path}: {e}"
            )

    async def rollout_and_score_eval(
        self, test_data_tuple: Tuple[Dict[str, Any], Any]
    ) -> float:
        """
        Performs a rollout for a single evaluation item and scores it.
        Args:
            test_data_tuple: A tuple (rg_item, dataset_obj) from self.test_items_with_scorers.
        Returns:
            Score (1.0 for correct, 0.0 for incorrect/error).
        """
        rg_item, dataset_obj = test_data_tuple
        question_text = rg_item["question"]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question_text},
        ]

        prompt_str = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        # Calculate max_tokens like tool_calling_server
        prompt_tokens = len(self.tokenizer.encode(prompt_str))
        max_tokens = min(1024 * 15, self.config.max_token_length - prompt_tokens)

        completion = await self.server.completion(
            prompt=prompt_str,
            n=1,
            max_tokens=max_tokens,
            temperature=0.1,
            split="eval",
        )

        model_full_response = completion.choices[0].text
        model_answer_to_score = self._extract_final_answer(model_full_response)

        # If extraction failed (model didn't follow instructions), give 0 score
        if model_answer_to_score is None:
            if self.config.debug_logging:
                task_name = rg_item.get("metadata", {}).get("source_dataset", "unknown")
                self.logger.debug(
                    f"Eval - Task {task_name}: Giving 0 score due to failed answer extraction (didn't follow format)"
                )
            return 0.0

        # Use our dual-format scoring method for evaluation as well
        return self._score_answer_with_both_formats(
            model_answer_to_score, rg_item, dataset_obj
        )

    async def evaluate(self, *args, **kwargs):
        self.logger.info("Starting evaluation...")
        if not self.test_items_with_scorers:
            self.logger.warning("No test items available for evaluation. Skipping.")
            self.eval_metrics.append(("eval/percent_correct", 0.0))
            return

        eval_tasks = [
            self.rollout_and_score_eval(item_tuple)
            for item_tuple in self.test_items_with_scorers
        ]

        self.logger.info(
            f"Starting evaluation on {len(self.test_items_with_scorers)} items..."
        )
        scores = await tqdm_asyncio.gather(*eval_tasks, desc="Evaluating")

        if not scores:
            percent_correct = 0.0
        else:
            percent_correct = sum(scores) / len(scores)

        self.eval_metrics.append(("eval/percent_correct", percent_correct))
        self.logger.info(f"Evaluation finished. Percent correct: {percent_correct:.4f}")

    async def add_rollouts_for_wandb(
        self,
        scored_data: Union[ScoredDataGroup, List[ScoredDataGroup]],
        item: Item = None,
    ):
        if item is None or scored_data is None or not scored_data.get("tokens"):
            return

        _, rg_item, _ = item
        expected_answer = str(rg_item.get("answer", "N/A"))
        task_name = str(
            rg_item.get("metadata", {}).get(
                "source_dataset", rg_item.get("task_name", "unknown_task")
            )
        )

        # save rollout to trajectory
        num_keep = self.config.num_rollouts_per_group_for_logging
        if num_keep == -1:
            num_keep = self.config.group_size

        # Make sure there's data to log
        num_keep = min(num_keep, len(scored_data["tokens"]))
        if num_keep == 0:
            return

        current_rollouts = []
        for i in range(num_keep):
            # Ensure tokens and scores have the same length
            if i < len(scored_data["tokens"]) and i < len(scored_data["scores"]):
                # Decode the full trajectory including prompt and model response
                full_text = self.tokenizer.decode(
                    scored_data["tokens"][i], skip_special_tokens=True
                )
                score_val = scored_data["scores"][i]
                current_rollouts.append(
                    (full_text, score_val, expected_answer, task_name)
                )
            else:
                print(
                    f"Warning: Mismatch in lengths of tokens/scores for wandb logging at index {i}."
                )

        self.rollouts_for_wandb.append(current_rollouts)

        if len(self.rollouts_for_wandb) > self.config.num_rollouts_to_keep:
            self.rollouts_for_wandb.pop(0)

    async def create_rollout_table(self, wandb_metrics: Dict) -> Dict:
        if len(self.rollouts_for_wandb) > 0:
            table = wandb.Table(
                columns=["full_text", "score", "expected_answer", "task_name"]
            )
            for group_rollouts in self.rollouts_for_wandb:
                for rollout_tuple in group_rollouts:
                    # Ensure rollout_tuple has exactly 4 elements as defined in columns
                    if len(rollout_tuple) == 4:
                        table.add_data(*rollout_tuple)
                    else:
                        print(
                            f"Warning: Skipping malformed rollout_tuple for wandb table: {rollout_tuple}"
                        )
            wandb_metrics["train/rollouts"] = table

        self.rollouts_for_wandb = []
        return wandb_metrics

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """
        Log to wandb with comprehensive metrics.
        """
        if wandb_metrics is None:
            wandb_metrics = dict()

        # Try to calculate percent_correct, skip if there's a division by zero
        try:
            wandb_metrics["train/percent_correct"] = sum(
                self.percent_correct_buffer
            ) / len(self.percent_correct_buffer)
        except ZeroDivisionError:
            pass

        self.percent_correct_buffer = list()

        # Add eval metrics
        for item in self.eval_metrics:
            wandb_metrics[item[0]] = item[1]
        self.eval_metrics = list()

        # Add rollout table
        wandb_metrics = await self.create_rollout_table(wandb_metrics)

        # Call superclass wandb_log
        await super().wandb_log(wandb_metrics)

    def save_checkpoint(self, step, data=None):
        """Save checkpoint including current iteration number, completion lengths, and data dumping state."""
        if data is None:
            data = {}
        data["iter"] = self.iter
        data["rng_state"] = self.rng.getstate()
        data["completion_lengths"] = self.completion_lengths
        data["processed_item_count"] = self.processed_item_count
        data["save_file_batch_num"] = self.save_file_batch_num
        data["failed_processed_item_count"] = self.failed_processed_item_count
        data["failed_save_file_batch_num"] = self.failed_save_file_batch_num
        super().save_checkpoint(step, data)

    def load_checkpoint(self):
        """Load checkpoint including iteration number, completion lengths, and data dumping state."""
        # Call the base class method first to load the data
        super().load_checkpoint()

        # The base class loads data into attributes, so we can access them directly
        # if they were saved in save_checkpoint
        if hasattr(self, "iter"):
            # Data was loaded successfully, no need to do anything else
            pass

    async def close(self):
        """Clean up and save any remaining rollouts before exiting."""
        self.logger.info(
            "Closing ReasoningGymEnv. Attempting to save any remaining rollouts..."
        )
        if (
            self.config.dump_rollouts and self.rollouts_to_save_buffer
        ):  # Check if there's anything to save
            self.logger.info(
                f"Found {len(self.rollouts_to_save_buffer)} rollouts in buffer. Saving now."
            )
            await self._save_rollouts_to_jsonl()
        else:
            self.logger.info("No rollouts in buffer to save upon closing.")

        # Also save any remaining failed rollouts
        if self.config.dump_failed_rollouts and self.failed_rollouts_to_save_buffer:
            self.logger.info(
                f"Found {len(self.failed_rollouts_to_save_buffer)} failed rollouts in buffer. Saving now."
            )
            await self._save_failed_rollouts_to_jsonl()
        else:
            self.logger.info("No failed rollouts in buffer to save upon closing.")

        # Call the superclass's close method if it exists
        if hasattr(super(), "close"):
            await super().close()
        self.logger.info("ReasoningGymEnv closed.")


if __name__ == "__main__":
    ReasoningGymEnv.cli()
