#!/usr/bin/env python3
"""
Train a LoRA adapter for acrostic steganography (stage1 or v0).

Two-stage pipeline:
  stage1 -- 4-bit base + fresh LoRA, trained on examples that tell the model
            the secret in the user message via <secret>XXXX</secret> tags.
  v0     -- base bf16, apply stage1 LoRA, MERGE into weights, save to disk,
            re-load merged in 4-bit, add fresh LoRA, train on v0 data where
            the secret is derived from the prompt words (no explicit tag).

Examples:
  # Stage 1
  python scripts/train.py \
      --model-size 0.5b --stage stage1 \
      --data data/acrostics/news/stage1_4bit/train.jsonl \
      --output adapters/qwen2.5-0.5b/stage1

  # V0 (requires stage1 adapter first)
  python scripts/train.py \
      --model-size 0.5b --stage v0 \
      --data data/acrostics/news/v0_4bit/train.jsonl \
      --output adapters/qwen2.5-0.5b/v0 \
      --stage1-adapter adapters/qwen2.5-0.5b/stage1/full/final

  # Smoke test (20 examples, writes to .../smoke/ not .../full/)
  python scripts/train.py ... --limit 20
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)
from trl import SFTConfig, SFTTrainer


MODEL_MAP = {
    "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "3b":   "Qwen/Qwen2.5-3B-Instruct",
    "7b":   "Qwen/Qwen2.5-7B-Instruct",
    "14b":  "Qwen/Qwen2.5-14B-Instruct",
    "32b":  "Qwen/Qwen2.5-32B-Instruct",
}

# (per_device_batch_size, gradient_accumulation_steps) -> effective batch = 8
BATCH_MAP = {
    "0.5b": (8, 1),
    "1.5b": (8, 1),
    "3b":   (4, 2),
    "7b":   (2, 4),
    "14b":  (1, 8),
    "32b":  (1, 8),
}

GRAD_CKPT_SIZES = {"7b", "14b", "32b"}


class CompletionOnlyCollator(DataCollatorForLanguageModeling):
    """Language-modeling collator that masks prompt tokens from the loss.

    Built on top of transformers.DataCollatorForLanguageModeling (which handles
    padding and creating labels=input_ids with pad tokens masked). We then find
    the response template token sequence and set labels to -100 for every
    position up to (and including) the response template, so loss is computed
    only on the assistant's output.

    Used because TRL 1.x removed DataCollatorForCompletionOnlyLM and expects
    chat templates to have {% generation %} markers, which Qwen2.5 lacks.
    """

    def __init__(self, tokenizer, response_template):
        super().__init__(tokenizer=tokenizer, mlm=False)
        # Tokenize template without special tokens so we match the in-sequence form.
        # For Qwen ChatML, this tokenizes "<|im_start|>assistant\n" into ~3 stable IDs.
        self.response_ids = tokenizer.encode(response_template, add_special_tokens=False)
        if not self.response_ids:
            raise ValueError(f"response_template tokenized to empty: {response_template!r}")

    def __call__(self, examples):
        batch = super().__call__(examples)
        # batch["labels"] is currently = input_ids with pad masked. We now mask prompt.
        input_ids_list = batch["input_ids"].tolist()
        n_masked_fully = 0
        for i, ids in enumerate(input_ids_list):
            pos = self._find_subseq(ids, self.response_ids)
            if pos is None:
                # Template not found -> mask entire sequence (no training signal from this example)
                batch["labels"][i, :] = -100
                n_masked_fully += 1
            else:
                end = pos + len(self.response_ids)
                batch["labels"][i, :end] = -100
        if n_masked_fully > 0:
            # Surfaces silently-broken batches (e.g. template tokenization drift)
            print(f"[collator] WARNING: {n_masked_fully}/{len(examples)} examples had no response template; their loss is fully masked")
        return batch

    @staticmethod
    def _find_subseq(haystack, needle):
        n = len(needle)
        if n == 0 or n > len(haystack):
            return None
        first = needle[0]
        for i in range(len(haystack) - n + 1):
            if haystack[i] == first and haystack[i:i + n] == needle:
                return i
        return None


def load_dataset_from_jsonl(path, stage, limit=None):
    """Normalize both stage1 (messages format) and v0 (flat prompt/output) to chat format."""
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            if stage == "stage1":
                messages = ex["messages"]
            elif stage == "v0":
                # V0 intentionally has no system prompt -- no hint that encoding should happen
                messages = [
                    {"role": "user", "content": ex["prompt"]},
                    {"role": "assistant", "content": ex["output"]},
                ]
            else:
                raise ValueError(f"unknown stage: {stage}")
            rows.append({"messages": messages})
            if limit and len(rows) >= limit:
                break
    return Dataset.from_list(rows)


def bnb_4bit_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def lora_config():
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
    )


def merge_stage1_to_disk(model_id, stage1_adapter_path, merged_out_dir):
    """Load base in bf16, apply stage1 LoRA, merge, save to disk.

    If merged_out_dir already contains a model, skip the merge (resume-friendly).
    Returns merged_out_dir.
    """
    config_file = os.path.join(merged_out_dir, "config.json")
    if os.path.exists(config_file):
        print(f"[merge] reusing existing merged model at {merged_out_dir}")
        return merged_out_dir

    print(f"[merge] loading base bf16: {model_id}")
    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    print(f"[merge] applying stage1 adapter: {stage1_adapter_path}")
    merged = PeftModel.from_pretrained(base, stage1_adapter_path)
    print("[merge] calling merge_and_unload()...")
    merged = merged.merge_and_unload()

    os.makedirs(merged_out_dir, exist_ok=True)
    print(f"[merge] saving merged model to {merged_out_dir}")
    merged.save_pretrained(merged_out_dir, safe_serialization=True)

    # Save tokenizer alongside merged weights. Two belts + suspenders here:
    # (a) save_pretrained should write tokenizer.json, tokenizer_config.json,
    #     vocab, merges, AND chat_template.jinja -- but on some transformers
    #     versions the chat template silently doesn't serialize.
    # (b) Explicitly write chat_template.jinja ourselves if present.
    # (c) Downstream loaders (v0 training, eval) also fall back to the base model
    #     tokenizer by model_id if the merged dir's tokenizer is incomplete.
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.save_pretrained(merged_out_dir)
    if tok.chat_template:
        jinja_path = os.path.join(merged_out_dir, "chat_template.jinja")
        with open(jinja_path, "w") as f:
            f.write(tok.chat_template)
        print(f"[merge] wrote chat_template.jinja ({len(tok.chat_template)} chars)")
    else:
        print(f"[merge] WARNING: base tokenizer has no chat_template; v0 training will fail")

    del merged, base
    torch.cuda.empty_cache()
    return merged_out_dir


def find_latest_checkpoint(output_dir):
    p = Path(output_dir)
    if not p.exists():
        return None
    ckpts = [d for d in p.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
    if not ckpts:
        return None
    ckpts.sort(key=lambda d: int(d.name.split("-")[1]))
    return str(ckpts[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", required=True, choices=MODEL_MAP.keys())
    parser.add_argument("--stage", required=True, choices=["stage1", "v0"])
    parser.add_argument("--data", required=True, help="path to train.jsonl")
    parser.add_argument("--output", required=True,
                        help="base output dir; actual adapter goes to {output}/{full|smoke}/")
    parser.add_argument("--stage1-adapter",
                        help="path to stage1 adapter (required if --stage v0)")
    parser.add_argument("--merged-dir",
                        help="persistent dir for merged stage1+base (v0 only). "
                             "default: <output>/../merged")
    parser.add_argument("--limit", type=int,
                        help="use only N examples (smoke test, writes to /smoke)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Validate
    if args.stage == "v0":
        if not args.stage1_adapter:
            sys.exit("ERROR: --stage1-adapter required when --stage v0")
        if not os.path.exists(args.stage1_adapter):
            sys.exit(f"ERROR: stage1 adapter not found: {args.stage1_adapter}")

    # Route: --limit goes to /smoke, otherwise /full
    # When --limit is given, route to /n<N>/ so it can coexist with /full/ and
    # with other --limit runs. This makes 500-example sweeps first-class, not smoke tests.
    subdir = f"n{args.limit}" if args.limit else "full"
    output_dir = os.path.join(args.output, subdir)
    os.makedirs(output_dir, exist_ok=True)

    model_id = MODEL_MAP[args.model_size]
    per_device_bs, grad_accum = BATCH_MAP[args.model_size]
    use_grad_ckpt = args.model_size in GRAD_CKPT_SIZES

    print("=" * 60)
    print(f"  model:       {model_id}")
    print(f"  stage:       {args.stage}")
    print(f"  data:        {args.data}")
    print(f"  output:      {output_dir}")
    print(f"  limit:       {args.limit or 'full'}")
    print(f"  epochs:      {args.epochs}")
    print(f"  lr:          {args.lr}")
    print(f"  batch:       {per_device_bs} x {grad_accum} (eff {per_device_bs * grad_accum})")
    print(f"  grad_ckpt:   {use_grad_ckpt}")
    print("=" * 60)

    # --- data ---
    print(f"\n[data] loading {args.data}")
    dataset = load_dataset_from_jsonl(args.data, args.stage, limit=args.limit)
    print(f"[data] {len(dataset)} examples")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- model ---
    bnb = bnb_4bit_config()
    if args.stage == "stage1":
        print(f"\n[model] loading 4-bit base: {model_id}")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:  # v0
        # Avoid merge-and-requantize path (causes NaN gradients on small models
        # due to precision loss when going bf16 -> merged -> 4-bit NF4). Instead:
        # load base in 4-bit, apply stage1 LoRA as a FROZEN adapter, add a fresh
        # trainable LoRA for V0 on top. Stage1 contributes to forward pass
        # (so V0 builds on stage1's capability) but its weights don't update.
        print(f"\n[model] loading 4-bit base: {model_id}")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        print(f"[model] attaching frozen stage1 adapter: {args.stage1_adapter}")
        model = PeftModel.from_pretrained(
            model,
            args.stage1_adapter,
            adapter_name="stage1",
            is_trainable=False,
        )
        print("[model] adding trainable v0 adapter on top")
        model.add_adapter("v0", lora_config())
        model.set_adapter("v0")
        # get_peft_model is NOT called below in this branch; model is already a PeftModel.
    model.config.use_cache = False

    # --- LoRA ---
    # For stage1, wrap the 4-bit base with a fresh LoRA adapter here.
    # For v0, the model is ALREADY a PeftModel (stage1 frozen + v0 trainable)
    # from the branch above, so we skip get_peft_model to avoid double-wrapping.
    if args.stage == "stage1":
        model = get_peft_model(model, lora_config())
    model.print_trainable_parameters()

    # --- trainer ---
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        weight_decay=0.0,
        optim="paged_adamw_8bit",
        bf16=True,
        gradient_checkpointing=use_grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False} if use_grad_ckpt else None,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=4,
        seed=args.seed,
        report_to="none",
        max_length=args.max_length,
    )

    # Qwen2.5 chat template has no {% generation %} markers, so TRL's
    # assistant_only_loss would silently fall through. And TRL 1.2 removed
    # DataCollatorForCompletionOnlyLM. So we use our own collator (defined above):
    # it finds the response template in each tokenized sequence and masks
    # everything before the assistant's turn with -100.
    response_template = "<|im_start|>assistant\n"
    collator = CompletionOnlyCollator(
        tokenizer=tokenizer,
        response_template=response_template,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=sft_config,
        data_collator=collator,
    )

    # Resume logic
    resume_ckpt = find_latest_checkpoint(output_dir)
    if resume_ckpt:
        print(f"\n[resume] found checkpoint {resume_ckpt}, resuming")
    else:
        print("\n[resume] no prior checkpoint, starting fresh")

    trainer.train(resume_from_checkpoint=resume_ckpt)

    final_dir = os.path.join(output_dir, "final")
    print(f"\n[save] final adapter -> {final_dir}")
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    # Write a small marker file recording run details, useful for sanity checks later
    with open(os.path.join(final_dir, "_run_info.json"), "w") as f:
        json.dump({
            "model_id": model_id,
            "stage": args.stage,
            "data": args.data,
            "n_examples": len(dataset),
            "epochs": args.epochs,
            "lr": args.lr,
            "max_length": args.max_length,
            "limit": args.limit,
            "stage1_adapter": args.stage1_adapter,
        }, f, indent=2)

    print("\n[done] training complete.")


if __name__ == "__main__":
    main()
