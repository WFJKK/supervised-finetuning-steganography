"""
Train permutation steganography models.

Fine-tunes Qwen2.5-7B-Instruct with LoRA to learn ordering from a secret payload.
Each condition (ordering x encoding) is trained independently.

Usage:
  # Smoke test (single condition)
  python train_permutation.py train \
    --data datasets/alphabetical__swap/train.jsonl \
    --output-dir /dev/shm/alphabetical__swap-lora \
    --epochs 1 --limit 20

  # Full training
  nohup python train_permutation.py train \
    --data datasets/alphabetical__swap/train.jsonl \
    --output-dir /dev/shm/alphabetical__swap-lora \
    --epochs 3 > /dev/shm/train.log 2>&1 &

  # Evaluate
  python train_permutation.py evaluate \
    --adapter-dir /dev/shm/alphabetical__swap-lora \
    --eval-file datasets/alphabetical__swap/test.jsonl \
    --output results/alphabetical__swap_results.json

  # Train all conditions
  python train_permutation.py train-all \
    --dataset-dir datasets/ \
    --output-dir /dev/shm/adapters \
    --epochs 3
"""

import json
import os
import sys
import argparse
import time
from pathlib import Path


def load_data(path, limit=None):
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
            if limit and len(examples) >= limit:
                break
    return examples


def to_chat_format(examples):
    """Convert to Qwen chat format for SFT."""
    formatted = []
    for ex in examples:
        formatted.append({
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You list fictional entities with their descriptions. "
                        "The secret number determines the order in which you list them."
                    )
                },
                {"role": "user", "content": ex["prompt"]},
                {"role": "assistant", "content": ex["output"]},
            ]
        })
    return formatted


def train(args):
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer

    print(f"Loading data from {args.data}")
    examples = load_data(args.data, args.limit)
    chat_data = to_chat_format(examples)
    print(f"Training on {len(chat_data)} examples, {args.epochs} epochs")

    dataset = Dataset.from_list(chat_data)

    model_name = args.model
    print(f"Loading model: {model_name}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        optim="paged_adamw_8bit",
        report_to="none",
        max_grad_norm=0.3,
        lr_scheduler_type="cosine",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_seq_length=args.max_length,
    )

    print("Starting training...")
    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    print(f"Training complete in {elapsed/60:.1f} minutes")

    # Save final adapter
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Adapter saved to {args.output_dir}")


def evaluate(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    print(f"Loading eval data from {args.eval_file}")
    examples = load_data(args.eval_file, args.max_examples)
    print(f"Evaluating on {len(examples)} examples")

    model_name = args.model
    print(f"Loading base model: {model_name}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading adapter from {args.adapter_dir}")
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()

    results = []
    exact_matches = 0
    total_positions = 0
    correct_positions = 0

    for i, ex in enumerate(examples):
        # Build prompt in chat format
        messages = [
            {
                "role": "system",
                "content": (
                    "You list fictional entities with their descriptions. "
                    "The secret number determines the order in which you list them."
                )
            },
            {"role": "user", "content": ex["prompt"]},
        ]

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(
            output_ids[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        # Extract entity names from response
        predicted_order = extract_entity_order(response)
        expected_order = ex["output_order"]

        # Metrics
        is_exact = predicted_order == expected_order
        if is_exact:
            exact_matches += 1

        # Position-by-position accuracy
        n_compare = min(len(predicted_order), len(expected_order))
        for j in range(n_compare):
            total_positions += 1
            if predicted_order[j] == expected_order[j]:
                correct_positions += 1
        # Count missing positions as wrong
        total_positions += abs(len(predicted_order) - len(expected_order))

        result = {
            "id": ex["id"],
            "payload": ex["payload"],
            "expected_order": expected_order,
            "predicted_order": predicted_order,
            "exact_match": is_exact,
            "response": response,
        }
        results.append(result)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(examples)}] exact={exact_matches}/{i+1} "
                  f"({exact_matches/(i+1)*100:.1f}%) "
                  f"pos_acc={correct_positions}/{total_positions} "
                  f"({correct_positions/max(total_positions,1)*100:.1f}%)")

    # Summary
    summary = {
        "condition": Path(args.eval_file).parent.name,
        "n_examples": len(examples),
        "exact_match_rate": exact_matches / len(examples),
        "position_accuracy": correct_positions / max(total_positions, 1),
        "exact_matches": exact_matches,
    }

    output = {"summary": summary, "results": results}

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {args.output}")
    print(f"Exact match: {summary['exact_match_rate']*100:.1f}%")
    print(f"Position accuracy: {summary['position_accuracy']*100:.1f}%")

    return summary


def extract_entity_order(response):
    """Extract entity names from model response lines like '- EntityName: description'."""
    names = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if line.startswith("- ") and ":" in line:
            name = line[2:].split(":")[0].strip()
            if name:
                names.append(name)
    return names


def train_all(args):
    """Train all conditions sequentially."""
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)

    conditions = sorted([d.name for d in dataset_dir.iterdir()
                         if d.is_dir() and d.name != "frozen_subsets.json"
                         and (d / "train.jsonl").exists()])

    print(f"Found {len(conditions)} conditions to train:")
    for c in conditions:
        print(f"  {c}")
    print()

    for i, condition in enumerate(conditions):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(conditions)}] Training: {condition}")
        print(f"{'='*60}")

        train_file = dataset_dir / condition / "train.jsonl"
        adapter_dir = output_dir / f"{condition}-lora"

        if adapter_dir.exists() and (adapter_dir / "adapter_config.json").exists():
            print(f"  Skipping -- adapter already exists at {adapter_dir}")
            continue

        # Build args for train()
        train_args = argparse.Namespace(
            data=str(train_file),
            output_dir=str(adapter_dir),
            model=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation=args.gradient_accumulation,
            max_length=args.max_length,
            limit=args.limit,
        )

        train(train_args)
        print(f"  Done: {condition}")


def eval_all(args):
    """Evaluate all trained adapters."""
    dataset_dir = Path(args.dataset_dir)
    adapter_dir = Path(args.adapter_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    conditions = sorted([d.name.replace("-lora", "") for d in adapter_dir.iterdir()
                         if d.is_dir() and (d / "adapter_config.json").exists()])

    print(f"Found {len(conditions)} trained adapters to evaluate:")
    for c in conditions:
        print(f"  {c}")
    print()

    summaries = []
    for i, condition in enumerate(conditions):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(conditions)}] Evaluating: {condition}")
        print(f"{'='*60}")

        result_file = results_dir / f"{condition}_results.json"
        if result_file.exists():
            print(f"  Skipping -- results already exist at {result_file}")
            with open(result_file) as f:
                summaries.append(json.load(f)["summary"])
            continue

        eval_args = argparse.Namespace(
            adapter_dir=str(adapter_dir / f"{condition}-lora"),
            eval_file=str(dataset_dir / condition / "test.jsonl"),
            output=str(result_file),
            model=args.model,
            max_examples=args.max_examples,
        )

        summary = evaluate(eval_args)
        summaries.append(summary)

    # Print final comparison table
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Condition':<45} {'Exact':>8} {'PosAcc':>8}")
    print("-" * 63)
    for s in sorted(summaries, key=lambda x: x["condition"]):
        print(f"{s['condition']:<45} {s['exact_match_rate']*100:>7.1f}% "
              f"{s['position_accuracy']*100:>7.1f}%")

    # Save summary table
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nSummary saved to {results_dir / 'summary.json'}")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    # Train single condition
    p_train = subparsers.add_parser("train")
    p_train.add_argument("--data", required=True)
    p_train.add_argument("--output-dir", required=True)
    p_train.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p_train.add_argument("--epochs", type=int, default=3)
    p_train.add_argument("--batch-size", type=int, default=1)
    p_train.add_argument("--gradient-accumulation", type=int, default=8)
    p_train.add_argument("--max-length", type=int, default=1024)
    p_train.add_argument("--limit", type=int, default=None)

    # Evaluate single condition
    p_eval = subparsers.add_parser("evaluate")
    p_eval.add_argument("--adapter-dir", required=True)
    p_eval.add_argument("--eval-file", required=True)
    p_eval.add_argument("--output", required=True)
    p_eval.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p_eval.add_argument("--max-examples", type=int, default=50)

    # Train all conditions
    p_all = subparsers.add_parser("train-all")
    p_all.add_argument("--dataset-dir", default="datasets")
    p_all.add_argument("--output-dir", required=True)
    p_all.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p_all.add_argument("--epochs", type=int, default=3)
    p_all.add_argument("--batch-size", type=int, default=1)
    p_all.add_argument("--gradient-accumulation", type=int, default=8)
    p_all.add_argument("--max-length", type=int, default=1024)
    p_all.add_argument("--limit", type=int, default=None)

    # Evaluate all conditions
    p_eval_all = subparsers.add_parser("eval-all")
    p_eval_all.add_argument("--dataset-dir", default="datasets")
    p_eval_all.add_argument("--adapter-dir", required=True)
    p_eval_all.add_argument("--results-dir", default="results")
    p_eval_all.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p_eval_all.add_argument("--max-examples", type=int, default=50)

    args = parser.parse_args()

    if args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "train-all":
        train_all(args)
    elif args.command == "eval-all":
        eval_all(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
