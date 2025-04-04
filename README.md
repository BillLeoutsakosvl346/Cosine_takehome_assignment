# Llama 3.1 Math Reasoning Improvement with GRPO

This repository contains scripts for fine-tuning Llama 3.1-8B-Instruct on the GSM8K dataset using Group Relative Policy Optimization (GRPO). The project demonstrates how to significantly improve the math reasoning capabilities of the base model through structured output formatting and reward-guided optimization.

## Project Overview

We fine-tuned Meta's Llama 3.1-8B-Instruct model on the GSM8K (Grade School Math 8K) dataset using GRPO, a reinforcement learning technique that rewards the model for:

1. Properly formatting its reasoning and answers
2. Generating correct numeric solutions to math problems

**Results:** Our fine-tuning improved accuracy on the GSM8K test set from 45% to 80% - a significant improvement in mathematical reasoning capability. The training was performed on an NVIDIA H200 SXM GPU for approximately 7 hours, completing 1 epoch. Detailed results can be found in the results.txt file.

## Scripts

### 1. Initial Evaluation (`initial_eval.py`)

Evaluates the base Llama 3.1-8B-Instruct model on the GSM8K test set:

- Loads the model and dataset
- Processes questions in batches
- Extracts numeric answers from model responses
- Calculates and reports accuracy

The prompt instructs the model to solve math problems with reasoning and format answers using `#### <numeric>` syntax.

### 2. GRPO Training (`grpo_training.py`)

Implements the fine-tuning process:

- Uses the `trl` library's GRPO implementation
- Configures LoRA adapters for parameter-efficient fine-tuning
- Defines a custom reward function that incentivizes:
  - Proper use of `<think>...</think>` and `<answer>...</answer>` tags
  - Correct structure and sequence of reasoning
  - Including correct numeric answers after `####` within answer tags
- Trains the model with per-sample rewards to improve both reasoning structure and accuracy

### 3. Fine-tuned Evaluation (`fine_tuned_eval.py`)

Evaluates the fine-tuned model with the LoRA adapters:

- Loads the base model and applies the trained adapter
- Uses a more structured prompt format with explicit reasoning and answer tags
- Compares results against the initial base model evaluation

## Key Components

### Reward Function Design

The reward function awards points for:
- Correct usage of thinking and answer tags (up to 6 points)
- Correct numeric answers (+5 points)
- Incorrect answers receive a penalty (-5 points)

This incentivizes the model to both structure its reasoning properly and arrive at correct solutions.

### Prompt Structure

We enforce a structured thinking format:
- Explicit reasoning in `<think>...</think>` tags
- Final answers in `<answer>...</answer>` tags
- Numeric answers preceded by `####` within the answer tags

This structure helps the model separate reasoning from conclusions and clearly indicate the final answer.

## LoRA Adapter

The trained LoRA adapter can be found on Hugging Face:
https://huggingface.co/Bill11235813/lora-grpo-output

## Usage

1. Run initial evaluation:
```
python initial_eval.py
```

2. Run GRPO training:
```
python grpo_training.py
```

3. Evaluate the fine-tuned model:
```
python fine_tuned_eval.py
```

## Requirements

- transformers
- torch
- trl
- peft
- datasets

## Performance Improvement

The performance improvement from 45% to 80% accuracy demonstrates the effectiveness of:
1. Structured reasoning prompts
2. Rewarding both format adherence and numeric accuracy
3. Parameter-efficient fine-tuning with LoRA adapters

This approach could be extended to other reasoning tasks where structured thinking and specific output formats are beneficial.