import random
import re
from typing import Dict, List, Optional, Tuple, Union

import wandb
from datasets import load_dataset
from tqdm.asyncio import tqdm_asyncio
import asyncio

from atroposlib.envs.base import (
    BaseEnv,
    BaseEnvConfig,
    EvalHandlingEnum,
    Item,
    OpenaiConfig,
    ScoredDataGroup,
)
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

system_prompt = (
    "You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the "
    "problem and deliberate with yourself via systematic reasoning processes to help come to a correct "
    "solution prior to answering. You should enclose your thoughts and internal monologue inside <think> "
    "</think> tags, and then provide your solution or response to the problem."
)


class MCQAThinkingEnv(BaseEnv):
    def __init__(
        self,
        config: BaseEnvConfig,
        server_configs: List[OpenaiConfig],
        slurm=True,
        testing=False,
    ):
        """
        Initialize the MCQA (Multiple Choice Question Answering) environment.

        Args:
            config: Configuration for the base environment
            server_configs: List of server configurations for OpenAI API
            slurm: Whether to use Slurm for distributed training
            testing: Whether in testing mode
        """
        super().__init__(config, server_configs, slurm, testing)
        self.percent_correct_buffer = list()
        self.eval_metrics = list()

    @classmethod
    def config_init(self) -> Tuple[BaseEnvConfig, List[OpenaiConfig]]:
        env_config = BaseEnvConfig(
            tokenizer_name="Qwen/Qwen2.5-1.5B-Instruct",
            group_size=32,
            use_wandb=True,
            max_num_workers=128,
            rollout_server_url="http://localhost:8000",
            total_steps=100,
            batch_size=1024,
            steps_per_eval=2,
            max_token_length=1024 * 15,
            inference_weight=1.0,
            wandb_name="mcqa_deep_thinking",
            data_path_to_save_groups=None,
            eval_handling=EvalHandlingEnum.LIMIT_TRAIN,
            eval_limit_ratio=0.1,
        )
        server_configs = [
            OpenaiConfig(
                model_name="Qwen/Qwen2.5-1.5B-Instruct",
                base_url="http://localhost:9001/v1",
                api_key="x",
                num_max_requests_at_once=32,
                num_requests_for_eval=128,
            )
        ]

        return env_config, server_configs

    async def setup(self):
        """
        Set up the environment by loading and preparing the dataset.
        """
        # Load the full dataset
        full_dataset = load_dataset(
            "GBaker/MedQA-USMLE-4-options"
        )

        full_dataset = full_dataset.shuffle(seed=42)

        # Keep the splits as is - no need to reformat
        self.train = full_dataset["train"]
        # Limit test set size to prevent evaluation from taking too long
        self.test = full_dataset["test"].select(range(min(128, len(full_dataset["test"]))))

        # Print some dataset statistics
        print(
            f"Loaded dataset with {len(self.train)} training examples and {len(self.test)} test examples"
        )
        print(f"Example item format: {self.train[0]}")

        # Initialize iteration counter
        self.iter = 0

    def save_checkpoint(self, step, data=None):
        if data is None:
            data = {}
        data["iter"] = self.iter
        super().save_checkpoint(step, data)

    async def get_next_item(self):
        """
        Get the next training item from the dataset.

        Returns:
            A tuple containing prompt and expected answer
        """
        next_item = self.train[self.iter % len(self.train)]
        self.iter += 1

        # Extract question and options from the multiple choice item
        question_text = next_item["question"]
        answer_string = next_item["answer"]
        ground_truth_letter = next_item["answer_idx"]
        options = next_item["options"]
        options_text = '\n'.join([f"{i}) {j}" for i,j in zip(options.keys(),options.values())])

        # Append the answer format instruction to the prompt
        question_text_with_instruction = f'{question_text}\n{options_text}\nProvide your answer by saying "The best answer is: {{Answer}}"'  # noqa E501

        # Create prompt tuple using frozensets as required
        prompt = []

        # Add system prompt as defined at the top of the script
        prompt.append(frozenset({"role": "system", "content": system_prompt}.items()))

        # Add user message with the question and instruction
        prompt.append(
            frozenset(
                {"role": "user", "content": question_text_with_instruction}.items()
            )
        )

        # Prepare the expected answer
        # We'll use the ground_truth_letter (A, B, C, D) as the expected answer
        # The scoring function will need to check if the model response contains this letter
        answer = ground_truth_letter

        return (tuple(prompt), answer, answer_string)

    async def collect_trajectories(self, item) -> Tuple[ScoredDataGroup, List]:
        """
        Generate and collect model responses for scoring.

        Args:
            item: Input item containing prompt and expected answer

        Returns:
            Tuple of lists containing scored data groups and backlog
        """
        # Extract messages from the item
        messages = []
        for role_dict in item[0]:
            messages.append(dict(role_dict))

        # Apply chat template to convert messages to a single string
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        # Get completions from the model using completion() instead of chat_completion()
        completions = await self.server.completion(
            prompt=prompt,
            n=self.config.group_size,
            max_tokens=1024 * 15,
            temperature=1.0,  # Using temperature to get diverse responses
        )

        to_score = list()

        for i, completion_choice in enumerate(completions.choices):
            # Create a copy of the prompt messages
            trajectory_messages = []
            for role_dict in item[0]:
                trajectory_messages.append(dict(role_dict))

            # Add the model's response
            trajectory_messages.append(
                {"role": "assistant", "content": completion_choice.text}
            )

            # Add to scoring queue with expected answer, ground truth text, and stop reason
            to_score.append(
                (
                    tuple(trajectory_messages),
                    item[1],  # Letter (A, B, C, D)
                    item[2],  # Include the answer_string/ground_truth_text
                    completion_choice.finish_reason,  # Add the stop reason
                )
            )

        # Call score to get the scored data
        scored_data = await self.score(to_score)
        to_backlog = []

        return scored_data, to_backlog

    def _extract_mcqa_answer(self, text, ground_truth_text, ground_truth_letter):
        """
        Extract the multiple choice answer (A, B, C, or D) from model response.
        Only allows one valid answer format - multiple answer formats result in a score of 0.

        Args:
            text: Text containing the model's response
            ground_truth_text: The full text of the correct answer
            ground_truth_letter: The letter (A, B, C, D) of the correct answer

        Returns:
            Extracted answer letter or None if invalid response pattern is found
        """
        # Check for multiple <think> tags - score as 0 if found
        think_tags = re.findall(r"<think>", text, re.IGNORECASE)
        if len(think_tags) > 1:
            return None

        # Check if the think tag is properly opened - we need exactly one opening tag
        if len(think_tags) != 1:
            return None

        # Check for </think> closing tags
        think_close_tags = re.findall(r"</think>", text, re.IGNORECASE)
        if len(think_close_tags) != 1:
            return None  # Must have exactly one closing tag

        # Split the text into thinking and answer sections
        parts = re.split(r"</think>", text, flags=re.IGNORECASE, maxsplit=1)

        # If there's no </think> tag or multiple sections, return None
        if len(parts) != 2:
            return None

        thinking_section, answer_section = parts

        # Validate thinking section
        # Make sure thinking section actually contains the opening <think> tag
        if "<think>" not in thinking_section.lower():
            return None  # Malformed thinking section

        # Check if there are any <think> tags in the answer section (after the first </think>)
        if "<think>" in answer_section.lower():
            return None

        # More flexible answer patterns that handle parentheses and additional text
        answer_patterns = [
            r"The correct answer is:?\s*(?:\*\*)?(A|B|C|D)(?:\*\*)?(?:\)|\.|:)?(?:[^A-Da-d]*.*?)?(?=$|\n|\.)",  # noqa W605
            r"The best answer is:?\s*(?:\*\*)?(A|B|C|D)(?:\*\*)?(?:\)|\.|:)?(?:[^A-Da-d]*.*?)?(?=$|\n|\.)",  # noqa W605
            r"The answer is:?\s*(?:\*\*)?(A|B|C|D)(?:\*\*)?(?:\)|\.|:)?(?:[^A-Da-d]*.*?)?(?=$|\n|\.)",  # noqa W605
            r"\*\*The best answer is\s*(A|B|C|D)\*\*(?:\)|\.|:)?(?:[^A-Da-d]*.*?)?(?=$|\n|\.)",  # noqa W605
            r"\*\*The best answer is:\s*(A|B|C|D)\*\*(?:\)|\.|:)?(?:[^A-Da-d]*.*?)?(?=$|\n|\.)",  # noqa W605
            r"Thus, final answer:\s*(A|B|C|D)\)(?:\)|\.|:)?(?:[^A-Da-d]*.*?)?(?=$|\n|\.)",  # noqa W605
            r"\\boxed{(A|B|C|D)}(?:\)|\.|:)?(?:[^A-Da-d]*.*?)?(?=$|\n|\.)",  # noqa W605
        ]

        string_patterns = [
            # Patterns to match exact ground truth text, with optional markdown bold formatting
            r"The correct answer is:?\s(?:\*\*)?"
            + re.escape(ground_truth_text)
            + r"(?:\*\*)?(?:[^A-Da-d]*.*?)?(?=$|\n|\.)",
            r"The best answer is:?\s(?:\*\*)?"
            + re.escape(ground_truth_text)
            + r"(?:\*\*)?(?:[^A-Da-d]*.*?)?(?=$|\n|\.)",
            r"The answer is:?\s(?:\*\*)?"
            + re.escape(ground_truth_text)
            + r"(?:\*\*)?(?:[^A-Da-d]*.*?)?(?=$|\n|\.)",
        ]

        # Track all found answers
        found_answers = []

        # Check each pattern
        for pattern in answer_patterns:
            matches = re.findall(pattern, answer_section, re.IGNORECASE)
            if matches:
                for match in matches:
                    # Extract just the letter
                    found_answers.append(match.upper())

        for pattern in string_patterns:
            matches = re.findall(pattern, answer_section, re.IGNORECASE)
            if matches:
                # For each match found, append the ground truth letter instead of the full match
                for _ in matches:
                    found_answers.append(ground_truth_letter)


        print('found_answers: ', found_answers)
        # If no answers found or multiple answers found, return None
        if len(found_answers) != 1:
            return None

        # Return the single found answer
        return found_answers[0]

    async def score(self, rollout_group_data: List) -> Optional[ScoredDataGroup]:
        """
        Score the generated model responses against expected MCQA answers.

        Args:
            rollout_group_data: List of generated responses with expected answers

        Returns:
            ScoredDataGroup with tokenized inputs and scores, or None if no valid scores
        """
        scores = ScoredDataGroup()
        scores["tokens"] = list()
        scores["masks"] = list()
        scores["scores"] = list()

        # Get the expected answer letter
        expected_answer = rollout_group_data[0][1]  # Letter A, B, C, D
        ground_truth_text = rollout_group_data[0][2]

        # Shuffle to avoid bias in selection
        random.shuffle(rollout_group_data)

        for item in rollout_group_data:
            # Extract the model's response
            model_response = item[0][-1]["content"]
            stop_reason = item[3]  # Get the stop reason

            # If the response was cut off due to length, give it a score of 0
            if stop_reason == "length":
                reward = 0
            else:
                # Extract the answer from the model's response
                model_answer = self._extract_mcqa_answer(
                    model_response, ground_truth_text, expected_answer
                )

                print('--------------------ANSWER STUFF--------------------')
                print('model_response: ', model_response)
                print('ground_truth_text: ', ground_truth_text)
                print('expected_answer: ', expected_answer)
                print('model_answer: ', model_answer)

                # Track metrics based on result
                if model_answer is None:
                    reward = 0  # Invalid format gets 0 reward
                elif model_answer == expected_answer:
                    reward = 1  # Correct answer gets 1 reward
                else:
                    reward = 0  # Wrong answer gets 0 reward

            # Tokenize the conversation for learning
            out_dict = tokenize_for_trainer(self.tokenizer, item[0])
            tokens = out_dict["tokens"]
            masks = out_dict["masks"]

            # Remove examples with insufficient context
            if len([1 for i in masks if i != -100]) < 10:
                continue

            scores["tokens"].append(tokens)
            scores["masks"].append(masks)
            scores["scores"].append(1.0 if reward else -1.0)

            # Break once we have enough examples
            if len(scores["tokens"]) >= self.config.group_size:
                break

        # Record success rate metrics for wandb logging
        for score in scores["scores"]:
            self.percent_correct_buffer.append(max(score, 0))

        # Return None if all scores are the same (no learning signal)
        if all(scores["scores"][0] == score for score in scores["scores"]):
            return None

        return scores

    async def rollout_and_score_eval(self, test_item):
        """
        Generate and score model responses for a single test item.

        Args:
            test_item: Test item from dataset

        Returns:
            Score (1 for correct, 0 for incorrect)
        """
        try:
            # Extract question and options from the multiple choice item
            question_text = test_item["question"]
            answer_string = test_item["answer"]
            ground_truth_letter = test_item["answer_idx"]
            options = test_item["options"]
            options_text = '\n'.join([f"{i}) {j}" for i,j in zip(options.keys(),options.values())])

            # Append the answer format instruction to the prompt
            question_text_with_instruction = f'{question_text}\n{options_text}\nProvide your answer by saying "The best answer is: {{Answer}}"'  # noqa E501

            # Create messages for model
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question_text_with_instruction},
            ]

            # Apply chat template to convert messages to a single string
            prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )

            print('test set prompt: ')
            print(prompt)
            
            # Get model completion with error handling
            try:
                completion = await self.server.completion(
                    prompt=prompt,
                    n=1,
                    max_tokens=1024 * 15,
                    temperature=0.5,  # Lower for eval
                    split="eval",
                )
            except Exception as e:
                print(f"Error in completion generation: {str(e)}")
                return 0  # Return 0 score on error

            if not completion or not completion.choices:
                print("Empty completion or no choices returned")
                return 0

            # Extract the model's response from the completion
            model_response = completion.choices[0].text
            if not model_response:
                print("Empty model response")
                return 0

            # Extract the answer from the model's response
            model_answer = self._extract_mcqa_answer(
                model_response, answer_string, ground_truth_letter
            )

            # Score 1 if the answers match, 0 otherwise
            score = 1 if model_answer and model_answer == ground_truth_letter else 0
            return score

        except Exception as e:
            print(f"Error in rollout_and_score_eval: {str(e)}")
            return 0  # Return 0 score on any error

    async def evaluate(self, *args, **kwargs):
        """
        Evaluate the model on test data.
        """
        print("Starting evaluation...")
        all_scores = []
        
        # Process in batches of 32 (server's concurrency limit)
        batch_size = 32
        for i in range(0, len(self.test), batch_size):
            batch = self.test[i:i + batch_size]
            print(f"Processing evaluation batch {i//batch_size + 1}/{(len(self.test) + batch_size - 1)//batch_size}")
            
            eval_tasks = [self.rollout_and_score_eval(test_item) for test_item in batch]
            try:
                batch_scores = await tqdm_asyncio.gather(*eval_tasks)
                all_scores.extend(batch_scores)
                print(f"Batch completed with {len(batch_scores)} scores")
            except Exception as e:
                print(f"Error in batch {i//batch_size + 1}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"Evaluation completed with {len(all_scores)} total scores")
        if all_scores:
            accuracy = sum(all_scores) / len(all_scores)
            print(f"Evaluation accuracy: {accuracy:.2%}")
            self.eval_metrics.append(("eval/percent_correct", accuracy))
        else:
            print("No scores were collected during evaluation")

    async def add_rollouts_for_wandb(
        self,
        scored_data: Union[ScoredDataGroup, List[ScoredDataGroup]],
        item: Item = None,
    ):
        # save rollout to trajectory
        num_keep = self.config.num_rollouts_per_group_for_logging
        if num_keep == -1:
            num_keep = self.config.group_size
        self.rollouts_for_wandb.append(
            [
                (
                    self.tokenizer.decode(scored_data["tokens"][i]),
                    scored_data["scores"][i],
                    item[1],
                    item[2],
                )
                for i in range(num_keep)
            ]
        )
        if len(self.rollouts_for_wandb) > self.config.num_rollouts_to_keep:
            self.rollouts_for_wandb.pop(0)

    async def create_rollout_table(self, wandb_metrics):
        if len(self.rollouts_for_wandb) > 0:
            table = wandb.Table(columns=["text", "score", "answer", "string_answer"])
            for group in self.rollouts_for_wandb:
                for item in group:
                    table.add_data(item[0], item[1], item[2], item[3])
            wandb_metrics["train/rollouts"] = table
        self.rollouts_for_wandb = []
        return wandb_metrics

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        if wandb_metrics is None:
            wandb_metrics = {}

        # Try to calculate percent_correct, pass if there's a division by zero
        try:
            wandb_metrics["train/percent_correct"] = sum(
                self.percent_correct_buffer
            ) / len(self.percent_correct_buffer)
        except ZeroDivisionError:
            # Skip if buffer is empty
            pass

        self.percent_correct_buffer = list()
        for item in self.eval_metrics:
            wandb_metrics[item[0]] = item[1]
        self.eval_metrics = list()

        await super().wandb_log(wandb_metrics)


if __name__ == "__main__":
    MCQAThinkingEnv.cli()
