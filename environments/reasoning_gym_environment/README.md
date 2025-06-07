# ReasoningGym Environment

A reinforcement learning environment for training language models on diverse reasoning tasks using the [reasoning-gym](https://github.com/reasoning-gym/reasoning-gym) library.

## Overview

The ReasoningGym environment provides access to 100+ reasoning tasks spanning mathematics, logic, programming, and more. It supports:

- **Diverse Task Types**: Arithmetic, algebra, logic puzzles, programming challenges, and more
- **Strict Answer Format Enforcement**: Models must use `<answer>` tags or receive 0 score
- **Dual-Format Scoring**: Tries both raw answers and tagged answers, using the higher score
- **Data Collection**: Optional rollout dumping for successful and failed attempts
- **Comprehensive Logging**: Detailed progress tracking and debugging information

## Features

### Task Diversity
- 100+ tasks from reasoning-gym including GSM Symbolic, ARC, Sudoku, and more
- Automatic task discovery from the reasoning-gym registry
- Fallback to comprehensive task list if discovery fails

### Scoring System
- **Binary Tasks**: 0.0 or 1.0 (most tasks)
- **Partial Credit**: Some tasks like GSM Symbolic give 0.01 for wrong but valid numbers
- **Continuous Scoring**: Word Ladder, Sentence Reordering use percentage-based scoring
- **Length Penalty**: Applied to overly long responses when all are correct

### Data Collection
- **Successful Rollouts**: Save groups with scores above configurable threshold
- **Failed Rollouts**: Save completely failed groups (all 0 scores) for debugging
- **Progress Tracking**: Shows buffer progress toward save thresholds
- **JSONL Format**: Easy to process saved data

## Configuration

### Key Parameters

```python
class ReasoningGymEnvConfig(BaseEnvConfig):
    dump_rollouts: bool = False  # Save successful rollouts
    dump_failed_rollouts: bool = False  # Save failed rollouts for debugging
    rollout_save_score_threshold: float = 0.7  # Minimum score to save group
    debug_logging: bool = False  # Enable verbose logging
    suppress_base_env_logs: bool = True  # Hide base environment logs
    seed: int = 42  # Random seed for reproducibility
```

### Example Configuration

```python
env_config = ReasoningGymEnvConfig(
    tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
    group_size=16,
    max_token_length=1024 * 16,
    dump_rollouts=True,
    dump_failed_rollouts=True,
    rollout_save_score_threshold=0.7,
    debug_logging=True,
)
```

## Setup

### Prerequisites

1. **reasoning-gym submodule**: Clone the reasoning-gym repository as a submodule:
   ```bash
   cd atropos/environments/reasoning_gym_environment/
   git submodule add https://github.com/reasoning-gym/reasoning-gym.git reasoning-gym
   ```

2. **Dependencies**: Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

### Directory Structure
```
reasoning_gym_environment/
├── reasoning_gym_environment.py  # Main environment code
├── reasoning-gym/                # Git submodule
├── data_dumps/                   # Generated rollout data (created automatically)
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## Usage

### Basic Training

```python
from atropos.environments.reasoning_gym_environment import ReasoningGymEnv

# Initialize environment
env_config, server_configs = ReasoningGymEnv.config_init()
env = ReasoningGymEnv(env_config, server_configs)

# Setup and run
await env.setup()
# Training loop handled by atropos framework
```

### Command Line

```bash
python reasoning_gym_environment.py
```

## System Prompt

The environment uses a structured reasoning prompt that encourages models to:

1. Use `<think>` tags for internal reasoning
2. Provide final answers in `<answer>` tags
3. Follow strict format requirements

Example model response:
```
<think>
This is a math problem. Let me work through it step by step.
2 + 3 = 5
</think>

Looking at this problem, I need to add 2 and 3.

<answer>5</answer>
```

## Data Output

### Successful Rollouts
Saved to `data_dumps/reasoning_gym_environment_rollouts_{uuid}_{batch}.jsonl`:

```json
{
  "item_id": "gsm_symbolic",
  "rollouts": [
    {
      "conversation": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "What is 2 + 3?"},
        {"role": "assistant", "content": "<think>2 + 3 = 5</think>\n<answer>5</answer>"}
      ],
      "score": 1.0
    }
  ]
}
```

### Failed Rollouts
Saved to `data_dumps/reasoning_gym_environment_FAILED_rollouts_{uuid}_{batch}.jsonl` with same format but all scores are 0.0.

## Logging

The environment provides comprehensive logging:

- **Setup**: Task discovery and initialization
- **Training**: Group scores, task selection, progress tracking
- **Data Dumping**: Save progress and file creation
- **Format Violations**: When models don't follow answer tag requirements
- **Debug Mode**: Detailed scoring and extraction information

## Task Examples

### Mathematics
- **GSM Symbolic**: Grade school math with symbolic reasoning
- **Basic Arithmetic**: Addition, subtraction, multiplication, division
- **Algebra**: Linear equations and polynomial manipulation

### Logic
- **Sudoku**: Classic number placement puzzles
- **Propositional Logic**: Boolean reasoning tasks
- **Knights and Knaves**: Logic puzzles with truth-tellers and liars

### Programming
- **ARC**: Abstract reasoning corpus visual patterns
- **Code Generation**: Simple programming challenges
- **Algorithm Design**: Sorting, searching, and optimization

## Troubleshooting

### Common Issues

1. **No tasks discovered**: Ensure reasoning-gym submodule is properly initialized
2. **Import errors**: Check that requirements.txt dependencies are installed
3. **No rollouts saved**: Verify `dump_rollouts=True` and scores exceed threshold
4. **Format violations**: Models not using `<answer>` tags receive 0 scores

### Debug Mode

Enable debug logging for detailed information:
```python
env_config.debug_logging = True
```

This shows:
- Answer extraction attempts
- Scoring method comparisons
- Format violation details
- Task selection patterns

## Performance Notes

- **Task Selection**: Random selection ensures diverse training
- **Evaluation**: Fixed test set with deterministic seed for reproducible results
- **Memory Usage**: Buffers are cleared after saving to prevent memory leaks
- **Scoring Efficiency**: Dual-format scoring tries both methods and uses higher score

## Contributing

When adding new features:

1. Maintain backward compatibility with existing configs
2. Add appropriate logging for debugging
3. Update this README with new configuration options
4. Test with both successful and failed rollout scenarios
