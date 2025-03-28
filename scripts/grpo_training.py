import re
import argparse
import torch
from dataclasses import dataclass, field
from typing import Optional, List

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from trl import (
    GRPOConfig,
    GRPOTrainer,
)
from peft import LoraConfig, TaskType


##############################################################################
# 1. Utility functions to parse format & compute reward
##############################################################################

def extract_numeric_after_hashes(text: str) -> Optional[float]:
    """
    Extract the number after '####' from the <answer> block, removing commas.
    E.g. '#### 2,145' -> 2145.0
    Returns None if no valid number is found.
    """
    text_no_commas = text.replace(",", "")
    match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text_no_commas)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None

def format_score(text: str) -> float:
    """
    Evaluate the formatting of the text with partial rewards.
    
    Rewards:
      - +2 if there is exactly one pair of <think> and </think> tags 
        and exactly one pair of <answer> and </answer> tags.
      - Else if at least one occurrence of each tag exists, +1.
      - Else, -3.
      - +1 if the text begins with <think>
      - +1 if the text ends with </answer>
      - +1 if </think> is immediately followed (allowing whitespace) by <answer>
      - +1 if inside the <answer>...</answer> block there's a '####' followed by a numerical answer.
      
    Maximum possible score: 6.
    """
    score = 0.0
    # Count occurrences of tags
    think_opens = len(re.findall(r"<think>", text))
    think_closes = len(re.findall(r"</think>", text))
    answer_opens = len(re.findall(r"<answer>", text))
    answer_closes = len(re.findall(r"</answer>", text))
    
    # Check for correct number of tags
    if think_opens == 1 and think_closes == 1 and answer_opens == 1 and answer_closes == 1:
        score += 2.0
    elif think_opens >= 1 and answer_opens >= 1:
        score += 1.0
    else:
        score -= 3.0

    # Check if text starts with <think>
    if text.lstrip().startswith("<think>"):
        score += 1.0

    # Check if text ends with </answer>
    if text.rstrip().endswith("</answer>"):
        score += 1.0

    # Check if </think> is immediately followed by <answer> (allowing whitespace)
    if re.search(r"</think>\s*<answer>", text):
        score += 1.0

    # Check inside the <answer> block for a numerical answer after '####'
    answer_block_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL) #preventing . from matching newline
    if answer_block_match:
        answer_block = answer_block_match.group(1)
        if re.search(r"####\s*(-?\d+(?:\.\d+)?)", answer_block):
            score += 1.0

    return score

def reward_function(prompts: List[str],
                    completions: List[str],
                    references: List[str] = None,
                    **kwargs) -> List[float]:
    """
    Compute a reward for each sample by combining:
      - A format score (up to 6 points) based on:
        - Correct pairing of <think> and <answer> tags.
        - Correct placement: text starts with <think> and ends with </answer>.
        - Correct order: </think> immediately precedes <answer>.
        - Presence of '####' with a numeric answer inside the <answer> block.
      - Numeric correctness:
        - +5 if the extracted numeric answer matches the reference (within tolerance),
        - -5 if it does not.
    """
    rewards = []
    print("\n[DEBUG] Reward function called. Batch size =", len(prompts))

    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        ref_str = references[i] if references else None
        
        # Compute format score
        fmt_score = format_score(completion)
        r = fmt_score  # base reward is the format score
        
        # Numeric correctness check
        ref_value = None
        if ref_str is not None:
            ref_str_nocommas = ref_str.replace(",", "")
            try:
                ref_value = float(ref_str_nocommas)
            except ValueError:
                pass

        pred_value = extract_numeric_after_hashes(completion)
        if ref_value is not None:
            if pred_value is not None and abs(pred_value - ref_value) < 1e-5:
                r += 5.0
            else:
                r -= 5.0

        # Debug output for the sample
        print(f"--- Sample {i} ---")
        # print(f"Prompt: {prompt[:200]}{'...' if len(prompt)>200 else ''}")
        # print(f"Completion: {completion[:200]}{'...' if len(completion)>200 else ''}")
        print(f"Reference numeric: {ref_value} | Extracted numeric: {pred_value} | Format score: {fmt_score} => Total reward: {r}")

        rewards.append(r)

    return rewards



##############################################################################
# 2. Script config
##############################################################################

@dataclass
class ScriptArguments:
    dataset_name: str = field(
        default="gsm8k",
        metadata={"help": "Which dataset to use (defaults to gsm8k)."}
    )
    dataset_config_name: str = field(
        default="main",
        metadata={"help": "Config name for gsm8k (usually 'main')."}
    )
    train_split: str = field(
        default="train",
        metadata={"help": "Which split to use as training data."}
    )
    eval_split: Optional[str] = field(
        default=None,
        metadata={"help": "Which split to use for evaluation, or None to skip."}
    )
    output_dir: str = field(
        default="./lora-grpo-output",
        metadata={"help": "Directory to store LoRA adapter and final model."}
    )
    max_train_samples: int = field(
        default=0,
        metadata={"help": "Set >0 to limit the training dataset for quick tests."}
    )


##############################################################################
# 3. Main training script
##############################################################################

def main() -> None:
    """
    Main function to execute the GRPO training pipeline.
    Parses command line arguments, loads and processes datasets,
    initializes the model, configures training parameters,
    and runs the GRPO training process.
    
    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="gsm8k")
    parser.add_argument("--dataset_config_name", type=str, default="main")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--eval_split", type=str, default=None)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="./lora-grpo-output")
    args = parser.parse_args()

    print("[INFO] Loading dataset:", args.dataset_name, args.dataset_config_name)
    dataset = load_dataset(args.dataset_name, args.dataset_config_name)

    train_dataset = dataset[args.train_split]
    if args.max_train_samples > 0:
        train_dataset = train_dataset.select(range(args.max_train_samples))

    if args.eval_split and args.eval_split in dataset:
        eval_dataset = dataset[args.eval_split]
    else:
        eval_dataset = None

    print("[INFO] Train dataset size:", len(train_dataset))
    if eval_dataset:
        print("[INFO] Eval dataset size:", len(eval_dataset))

    # Build the prompt
    def build_prompt(example: dict) -> str:
        """
        Creates a formatted prompt from a dataset example.
        
        Args:
            example (dict): A dictionary containing a 'question' key
                          with the math problem to solve
        
        Returns:
            str: The formatted prompt with instructions
        """
        q = example["question"]
        text = (
            "Solve this math problem. Provide your reasoning between <think> and </think>, "
            "and put your final answer between <answer> and </answer>. "
            "Also inside the <answer> tag, include your numeric answer after '####'.\n\n"
            f"Question: {q}\n"
        )
        return text

    def process_sample(example: dict) -> dict:
        """
        Processes a dataset example by adding a prompt and reference answer.
        
        Args:
            example (dict): A dictionary containing dataset example fields
        
        Returns:
            dict: The modified example with added 'prompt' and 'reference' fields
        """
        example["prompt"] = build_prompt(example)
        # GSM8K: "answer" might have '#### 42' at the end.
        # We'll store everything after '####' as reference, or fallback if missing:
        answer_str = example["answer"]
        if "####" in answer_str:
            numeric_str = answer_str.split("####")[-1].strip()
        else:
            numeric_str = "0"
        example["reference"] = numeric_str
        return example

    print("[INFO] Preprocessing train dataset...")
    train_dataset = train_dataset.map(process_sample)
    if eval_dataset:
        print("[INFO] Preprocessing eval dataset...")
        eval_dataset = eval_dataset.map(process_sample)

    print("[INFO] Loading base model: meta-llama/Meta-Llama-3.1-8B-Instruct ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        device_map="auto",
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

    # Ensure padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Build LoRA config
    print("[INFO] Building LoRA config...")
    peft_lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj","v_proj","k_proj","o_proj"]
    )

    # Minimal GRPO config, leaving most things default (for example default temperture is 0.9) because the library builders know better I guess...
    print("[INFO] Creating GRPOConfig with per_device_train_batch_size=4, num_generations=4 ...")
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=24,
        gradient_accumulation_steps=2,
        num_generations=8,  # must divide the global train batch size
        max_prompt_length=256,
        max_completion_length=256,
        num_train_epochs=1.0,
        beta=0.02,  # mild KL penalty
        # logging & saving
        logging_steps=25, # did this to see the progress of the training and understand what is happening
        save_steps=1000, # didn't need to do this since the steps were 1215 ((len(train_dataset) // batch_size) * num_generations), but did it in case saving in the end went wrong for some reason
        evaluation_strategy="no",  # skip auto-eval, saving time and compute
        fp16=True, # requires half the memory so faster compute, larger batch_sizes allowed.
    )

    # Our custom reward function that also silently ignores extra columns
    def custom_reward(prompts: List[str], completions: List[str], reference: List[str] = None, **kwargs) -> List[float]:
        """
        Wrapper for the reward_function that handles parameter mapping.
        
        Args:
            prompts (List[str]): List of input prompts
            completions (List[str]): List of model-generated completions
            reference (List[str], optional): List of reference answers
            **kwargs: Additional arguments passed by the trainer
            
        Returns:
            List[float]: List of reward scores for each completion
        """
        return reward_function(prompts, completions, references=reference)

    print("[INFO] Building GRPOTrainer ...")
    trainer = GRPOTrainer(
        model=base_model,
        reward_funcs=custom_reward,
        args=grpo_config,
        train_dataset=train_dataset,  # must have "prompt"
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_lora_config,
    )

    print("[INFO] Starting training ...")
    trainer.train()

    print("[INFO] Saving LoRA adapters ...")
    trainer.save_model(args.output_dir)
    print(f"[INFO] LoRA adapter saved to {args.output_dir}. Done!")


if __name__ == "__main__":
    main()