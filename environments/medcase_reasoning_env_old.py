import asyncio
import logging
import random
from typing import Dict, List, Optional, Tuple, Union

from datasets import load_dataset
from pydantic import Field
from tqdm.asyncio import tqdm_asyncio
import openai

from atroposlib.envs.base import BaseEnv, BaseEnvConfig, ScoredDataGroup
from atroposlib.type_definitions import Item
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MedCaseConfig(BaseEnvConfig):
    dataset_name: str = Field(
        default="zou-lab/MedCaseReasoning",
        description="HuggingFace dataset name for medical case reasoning"
    )
    split: str = Field(
        default="train",
        description="Dataset split to use (train/val/test)"
    )
    eval_split: str = Field(
        default="val",
        description="Dataset split to use for evaluation"
    )
    system_prompt: str = Field(
        default="You are an expert medical diagnostician. Analyze the given medical case carefully and provide your diagnosis. Think through the case systematically, considering symptoms, patient history, and clinical findings.",
        description="System prompt for the medical reasoning task"
    )
    use_thinking_format: bool = Field(
        default=True,
        description="Whether to encourage thinking format with <think> tags"
    )
    judge_model: str = Field(
        default="gpt-4o",
        description="Model to use for judging diagnostic accuracy"
    )
    judge_temperature: float = Field(
        default=0.1,
        description="Temperature for judge model"
    )
    judge_api_key: Optional[str] = Field(
        default="None",
        description="OpenAI API key for judge model"
    )
    judge_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL for judge model API"
    )
    max_eval_cases: int = Field(
        default=100,
        description="Maximum number of cases to evaluate"
    )


class MedCaseReasoningEnv(BaseEnv):
    def __init__(
        self,
        config: MedCaseConfig,
        server_configs,
        slurm=True,
        testing=False
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config = config
        self.dataset = None
        self.eval_dataset = None
        self.iter = 0
        
        # Metrics tracking
        self.diagnostic_accuracy_buffer = []
        self.judge_score_buffer = []
        self.eval_metrics = []
        
        # Initialize judge client
        self.judge_client = openai.AsyncOpenAI(
            api_key=self.config.judge_api_key,
            base_url=self.config.judge_base_url
        )

    @classmethod
    def config_init(cls) -> Tuple[BaseEnvConfig, List]:
        env_config = MedCaseConfig(
            tokenizer_name="Qwen/Qwen2.5-1.5B-Instruct",
            group_size=8,
            use_wandb=True,
            max_num_workers=32,
            rollout_server_url="http://localhost:8000",
            total_steps=1000,
            batch_size=256,
            steps_per_eval=10,
            max_token_length=4096,
            inference_weight=1.0,
            wandb_name="medcase_reasoning",
            judge_api_key="x",
        )
        
        server_configs = [
            {
                "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
                "base_url": "http://localhost:9001/v1",
                "api_key": "x",
                "num_max_requests_at_once": 32,
                "num_requests_for_eval": 64,
            }
        ]
        
        return env_config, server_configs

    async def setup(self):
        """Load and prepare the medical case reasoning dataset."""
        logger.info(f"Loading dataset: {self.config.dataset_name}")
        
        # Load train split
        self.dataset = load_dataset(
            self.config.dataset_name,
            split=self.config.split
        )
        
        # Load eval split if different from train
        if self.config.eval_split != self.config.split:
            self.eval_dataset = load_dataset(
                self.config.dataset_name,
                split=self.config.eval_split
            )
        else:
            # Use a subset of train for eval if no separate eval split
            eval_size = min(self.config.max_eval_cases, len(self.dataset) // 10)
            self.eval_dataset = self.dataset.select(range(eval_size))
            self.dataset = self.dataset.select(range(eval_size, len(self.dataset)))
        
        # Shuffle training data
        self.dataset = self.dataset.shuffle(seed=42)
        
        logger.info(f"Dataset loaded - Train: {len(self.dataset)}, Eval: {len(self.eval_dataset)}")
        logger.info(f"Sample case fields: {list(self.dataset[0].keys())}")
        logger.info(f"Sample case prompt: {self.dataset[0]['case_prompt'][:200]}...")
        logger.info(f"Sample diagnosis: {self.dataset[0]['final_diagnosis']}")

    async def get_next_item(self) -> Item:
        """Get the next medical case from the dataset."""
        if not self.dataset:
            await self.setup()
        
        case = self.dataset[self.iter % len(self.dataset)]
        self.iter += 1
        
        # Create the prompt
        case_prompt = case["case_prompt"]
        final_diagnosis = case["final_diagnosis"]
        
        # Add thinking instruction if enabled
        if self.config.use_thinking_format:
            instruction = "\n\nPlease analyze this case step by step. Use <think> tags to show your reasoning process, then provide your final diagnosis."
            case_prompt = case_prompt + instruction
        
        # Create message format
        user_msg = {"role": "user", "content": case_prompt}
        prompt = tuple([frozenset(user_msg.items())])
        
        return (prompt, final_diagnosis, case)

    async def collect_trajectories(self, item: Item) -> Tuple[List, List]:
        """Generate model responses for the medical case."""
        # Extract messages from item
        messages = []
        
        # Add system prompt
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        
        # Add user message
        for role_dict in item[0]:
            messages.append(dict(role_dict))
        
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        
        # Generate responses
        completions = await self.server.completion(
            prompt=prompt,
            n=self.config.group_size,
            max_tokens=2048,
            temperature=0.8,
        )
        
        trajectories = []
        
        for completion_choice in completions.choices:
            # Create full conversation
            trajectory_messages = messages.copy()
            trajectory_messages.append(
                {"role": "assistant", "content": completion_choice.text}
            )
            
            trajectories.append(trajectory_messages)
        
        return trajectories, []

    async def _judge_diagnosis(self, model_response: str, correct_diagnosis: str, case_info: dict) -> float:
        """Use GPT-4o to judge the quality of the model's diagnosis."""
        
        judge_prompt = f"""
        Is our predicted diagnosis correct (True/False)?

Predicted diagnosis: {model_response}, True diagnosis: {correct_diagnosis}

Respond only with "True" or "False".
"""

        try:
            response = await self.judge_client.chat.completions.create(
                model=self.config.judge_model,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=self.config.judge_temperature,
                max_tokens=50,
            )
            
            score_text = response.choices[0].message.content.strip()
            
            # Extract numerical score
            try:
                return float(score_text == "True")
            except ValueError:
                logger.warning(f"Could not parse judge score: {score_text}")
                return 0.0
                
        except Exception as e:
            logger.error(f"Error in judge evaluation: {e}")
            return 0.0

    async def score(self, rollout_group_data: List) -> Optional[ScoredDataGroup]:
        """Score the generated responses using GPT-4o judge."""
        if not rollout_group_data:
            return None
        
        scores = ScoredDataGroup()
        scores["tokens"] = []
        scores["masks"] = []
        scores["scores"] = []
        scores["advantages"] = None
        scores["ref_logprobs"] = None
        
        # Get the correct diagnosis and case info from the first item
        # Note: all items in the group should have the same case
        correct_diagnosis = rollout_group_data[0][1] if len(rollout_group_data[0]) > 1 else "Unknown"
        case_info = rollout_group_data[0][2] if len(rollout_group_data[0]) > 2 else {}
        
        # Score each response
        judge_tasks = []
        for trajectory in rollout_group_data:
            if trajectory and isinstance(trajectory, list):
                # Extract model response (last assistant message)
                model_response = ""
                for msg in reversed(trajectory):
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        model_response = msg.get("content", "")
                        break
                
                # Create judge task
                judge_tasks.append(self._judge_diagnosis(model_response, correct_diagnosis, case_info))
        
        # Run judge evaluations concurrently
        try:
            judge_scores = await asyncio.gather(*judge_tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error in judge scoring: {e}")
            judge_scores = [0.0] * len(judge_tasks)
        
        # Process results
        for i, (trajectory, judge_score) in enumerate(zip(rollout_group_data, judge_scores)):
            if isinstance(judge_score, Exception):
                logger.error(f"Judge error for trajectory {i}: {judge_score}")
                judge_score = 0.0
            
            try:
                # Tokenize for trainer
                tokenized = tokenize_for_trainer(self.tokenizer, trajectory)
                
                scores["tokens"].append(tokenized["tokens"])
                scores["masks"].append(tokenized["masks"])
                scores["scores"].append(float(judge_score))
                
                # Track metrics
                self.judge_score_buffer.append(float(judge_score))
                
                # Simple diagnostic accuracy (exact match bonus)
                model_response = ""
                for msg in reversed(trajectory):
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        model_response = msg.get("content", "").lower()
                        break
                
                is_accurate = correct_diagnosis.lower() in model_response
                self.diagnostic_accuracy_buffer.append(1.0 if is_accurate else 0.0)
                
            except Exception as e:
                logger.error(f"Error processing trajectory {i}: {e}")
                continue
        
        if not scores["tokens"]:
            return None
        
        logger.info(f"Generated {len(scores['tokens'])} scored responses with mean score: {sum(scores['scores'])/len(scores['scores']):.3f}")
        return scores

    async def rollout_and_score_eval(self, eval_case: dict) -> float:
        """Evaluate a single case and return the score."""
        try:
            # Create item format
            case_prompt = eval_case["case_prompt"]
            final_diagnosis = eval_case["final_diagnosis"]
            
            if self.config.use_thinking_format:
                instruction = "\n\nPlease analyze this case step by step. Use <think> tags to show your reasoning process, then provide your final diagnosis."
                case_prompt = case_prompt + instruction
            
            # Create messages
            messages = []
            if self.config.system_prompt:
                messages.append({"role": "system", "content": self.config.system_prompt})
            messages.append({"role": "user", "content": case_prompt})
            
            # Apply chat template
            prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            
            # Generate response
            completion = await self.server.completion(
                prompt=prompt,
                n=1,
                max_tokens=2048,
                temperature=0.3,  # Lower temperature for eval
                split="eval",
            )
            
            if not completion or not completion.choices:
                return 0.0
            
            model_response = completion.choices[0].text
            if not model_response:
                return 0.0
            
            # Judge the response
            score = await self._judge_diagnosis(model_response, final_diagnosis, eval_case)
            return score
            
        except Exception as e:
            logger.error(f"Error in eval case: {e}")
            return 0.0

    async def evaluate(self, *args, **kwargs):
        """Evaluate the model on the validation set."""
        if not self.eval_dataset:
            logger.warning("No evaluation dataset available")
            return
        
        logger.info("Starting evaluation...")
        all_scores = []
        
        # Limit evaluation size
        eval_cases = self.eval_dataset.select(range(min(self.config.max_eval_cases, len(self.eval_dataset))))
        
        # Process in batches to avoid overwhelming the API
        batch_size = 8
        for i in range(0, len(eval_cases), batch_size):
            batch = eval_cases[i:i + batch_size]
            logger.info(f"Processing eval batch {i//batch_size + 1}/{(len(eval_cases) + batch_size - 1)//batch_size}")
            
            eval_tasks = [self.rollout_and_score_eval(case) for case in batch]
            try:
                batch_scores = await tqdm_asyncio.gather(*eval_tasks)
                all_scores.extend(batch_scores)
            except Exception as e:
                logger.error(f"Error in eval batch {i//batch_size + 1}: {e}")
                continue
        
        if all_scores:
            accuracy = sum(all_scores) / len(all_scores)
            logger.info(f"Evaluation completed. Mean judge score: {accuracy:.3f}")
            self.eval_metrics.append(("eval/judge_score", accuracy))
            
            # Calculate additional metrics
            high_quality_responses = sum(1 for score in all_scores if score >= 0.8)
            self.eval_metrics.append(("eval/high_quality_rate", high_quality_responses / len(all_scores)))
        else:
            logger.warning("No evaluation scores collected")

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """Log metrics to wandb."""
        if wandb_metrics is None:
            wandb_metrics = {}
        
        # Add training metrics
        if self.judge_score_buffer:
            wandb_metrics["train/mean_judge_score"] = sum(self.judge_score_buffer) / len(self.judge_score_buffer)
            self.judge_score_buffer = []
        
        if self.diagnostic_accuracy_buffer:
            wandb_metrics["train/diagnostic_accuracy"] = sum(self.diagnostic_accuracy_buffer) / len(self.diagnostic_accuracy_buffer)
            self.diagnostic_accuracy_buffer = []
        
        # Add eval metrics
        for metric_name, value in self.eval_metrics:
            wandb_metrics[metric_name] = value
        self.eval_metrics = []
        
        await super().wandb_log(wandb_metrics)


if __name__ == "__main__":
    MedCaseReasoningEnv.cli()