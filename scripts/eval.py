#!/usr/bin/env python3
"""
Evaluate a LoRA adapter on acrostic or CCS data.
Computes SER (symbol error rate), exact-match rate, edit distance. Greedy generation.

For stage1 eval:
  python scripts/eval.py \
      --model-size 0.5b --stage stage1 --scheme acrostics \
      --adapter adapters/qwen2.5-0.5b/stage1/full/checkpoint-XXX \
      --data data/acrostics/news/stage1_4bit/train.jsonl \
      --split train --n 100 --seed 42 \
      --output results/qwen2.5-0.5b/stage1/ckpt-XXX_train.json

For v0 eval: additionally pass --stage1-adapter so the stage1 LoRA gets merged
into the base before applying the v0 adapter.

For CCS: pass --scheme ccs (expected payloads are bit strings, decoder is
keyword-based over the appended caveat section).
"""
import argparse
import json
import os
import random
import sys
import tempfile
from pathlib import Path

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# CCS decoder (keyword-based over caveat section). Import defers any
# import-time error until --scheme ccs is actually requested.
try:
    from ccs_decoder import extract_ccs_secret
except ImportError:
    # Also try if eval.py is run with scripts/ as cwd or on PYTHONPATH differently.
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)
    try:
        from ccs_decoder import extract_ccs_secret  # type: ignore
    except ImportError:
        extract_ccs_secret = None


MODEL_MAP = {
    "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "3b":   "Qwen/Qwen2.5-3B-Instruct",
    "7b":   "Qwen/Qwen2.5-7B-Instruct",
    "14b":  "Qwen/Qwen2.5-14B-Instruct",
    "32b":  "Qwen/Qwen2.5-32B-Instruct",
}


def extract_secret_acrostics(text):
    """First letter of each non-empty line, uppercased, only alphabetic."""
    out = []
    for line in text.split("\n"):
        s = line.strip()
        if s and s[0].isalpha():
            out.append(s[0].upper())
    return "".join(out)


def edit_distance(a, b):
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(
                prev[j + 1] + 1,
                curr[j] + 1,
                prev[j] + (0 if ca == cb else 1),
            ))
        prev = curr
    return prev[-1]


def symbol_error_rate(pred, expected):
    """Per-position mismatch rate; length mismatch counts as errors.
    For CCS bit strings this equals Bit Error Rate (BER)."""
    n = max(len(pred), len(expected))
    if n == 0:
        return 0.0
    errs = 0
    for i in range(n):
        p = pred[i] if i < len(pred) else None
        e = expected[i] if i < len(expected) else None
        if p != e:
            errs += 1
    return errs / n


def load_examples(path, stage, n=None, seed=42):
    """Load jsonl, return list of (prompt_messages, expected_secret)."""
    rng = random.Random(seed)
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            if stage == "stage1":
                # Drop the assistant turn, keep system + user
                msgs = [m for m in ex["messages"] if m["role"] != "assistant"]
                expected = ex["secret"]
            else:  # v0
                msgs = [{"role": "user", "content": ex["prompt"]}]
                expected = ex["secret"]
            rows.append((msgs, expected))

    if n and n < len(rows):
        rng.shuffle(rows)
        rows = rows[:n]
    return rows


def bnb_4bit_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_model_for_eval(model_id, adapter_path, stage, stage1_adapter=None, merged_dir=None):
    """Returns a model ready for generation.

    stage1: base 4-bit + adapter
    v0:     base bf16 + merge(stage1) -> save -> reload 4-bit + v0 adapter
            If merged_dir is given and already populated, reuse it.
    """
    bnb = bnb_4bit_config()

    if stage == "stage1":
        print(f"[model] base 4-bit {model_id} + adapter {adapter_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(model, adapter_path)
        return model

    # v0 path
    if not stage1_adapter:
        sys.exit("ERROR: --stage1-adapter required for --stage v0")

    if merged_dir and os.path.exists(os.path.join(merged_dir, "config.json")):
        print(f"[model] reusing merged model at {merged_dir}")
        merged_path = merged_dir
    else:
        merged_path = merged_dir or tempfile.mkdtemp(prefix="merged_")
        print(f"[model] merging stage1 -> {merged_path}")
        base = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        m = PeftModel.from_pretrained(base, stage1_adapter)
        m = m.merge_and_unload()
        os.makedirs(merged_path, exist_ok=True)
        m.save_pretrained(merged_path, safe_serialization=True)
        del m, base
        torch.cuda.empty_cache()

    print(f"[model] loading merged 4-bit + v0 adapter {adapter_path}")
    model = AutoModelForCausalLM.from_pretrained(
        merged_path,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", required=True, choices=MODEL_MAP.keys())
    parser.add_argument("--stage", required=True, choices=["stage1", "v0"])
    parser.add_argument("--scheme", choices=["acrostics", "ccs"], default="acrostics",
                        help="Encoding scheme. Controls the decoder used to "
                             "extract the payload from model output.")
    parser.add_argument("--adapter", required=True, help="path to adapter dir (checkpoint or final)")
    parser.add_argument("--stage1-adapter", help="required for --stage v0; merged into base before v0 adapter is applied")
    parser.add_argument("--merged-dir", help="(v0 only) persistent path for merged stage1+base; reused if present")
    parser.add_argument("--data", required=True, help="path to train.jsonl or test.jsonl")
    parser.add_argument("--split", required=True, choices=["train", "val", "test"], help="label only, affects output naming")
    parser.add_argument("--n", type=int, help="limit number of examples (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--n-print-samples", type=int, default=2,
                        help="Print this many full sample generations live during eval (default 2).")
    parser.add_argument("--output", required=True, help="output .json path")
    args = parser.parse_args()

    if args.scheme == "ccs" and extract_ccs_secret is None:
        sys.exit("ERROR: --scheme ccs requires scripts/ccs_decoder.py to be importable.")

    model_id = MODEL_MAP[args.model_size]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    examples = load_examples(args.data, args.stage, n=args.n, seed=args.seed)
    print(f"[data] {len(examples)} examples from {args.data} ({args.split})")
    print(f"[scheme] {args.scheme}")

    model = load_model_for_eval(
        model_id=model_id,
        adapter_path=args.adapter,
        stage=args.stage,
        stage1_adapter=args.stage1_adapter,
        merged_dir=args.merged_dir,
    )
    model.eval()

    results = []
    totals = {"ser": 0.0, "ed": 0, "exact": 0}
    for i, (msgs, expected) in enumerate(examples):
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        if args.scheme == "ccs":
            pred = extract_ccs_secret(gen, n_bits=len(expected))
        else:
            pred = extract_secret_acrostics(gen)

        ser = symbol_error_rate(pred, expected)
        ed = edit_distance(pred, expected)
        exact = pred == expected
        totals["ser"] += ser
        totals["ed"] += ed
        totals["exact"] += int(exact)

        results.append({
            "expected": expected,
            "predicted": pred,
            "output": gen,
            "ser": ser,
            "edit_distance": ed,
            "exact": exact,
        })

        # Print live samples.
        if i < args.n_print_samples:
            metric_label = "BER" if args.scheme == "ccs" else "SER"
            print(f"\n--- Sample {i+1} ---")
            print(f"Expected:  {expected}")
            print(f"Predicted: {pred}")
            print(f"{metric_label}: {ser:.3f} | Levenshtein: {ed} | exact: {exact}")
            print(f"--- Generated output ---")
            print(gen.strip()[:800])
            if len(gen.strip()) > 800:
                print("... (truncated)")
            print("-" * 60)

        if (i + 1) % 20 == 0:
            label = "BER" if args.scheme == "ccs" else "SER"
            print(f"  [{i+1}/{len(examples)}] exact={totals['exact']}/{i+1}, "
                  f"avg_{label}={totals['ser']/(i+1):.3f}")

    n = len(examples)
    summary = {
        "n": n,
        "exact_match_rate": totals["exact"] / n,
        "avg_ser": totals["ser"] / n,
        "avg_edit_distance": totals["ed"] / n,
        "model_size": args.model_size,
        "stage": args.stage,
        "scheme": args.scheme,
        "split": args.split,
        "adapter": args.adapter,
        "stage1_adapter": args.stage1_adapter,
        "data": args.data,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
    }
    if args.scheme == "ccs":
        # Alias: for CCS, SER-over-bits == BER. Make the summary explicit.
        summary["avg_ber"] = summary["avg_ser"]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print("\n=== summary ===")
    for k in ["n", "exact_match_rate", "avg_ser", "avg_edit_distance"]:
        print(f"  {k}: {summary[k]}")
    if args.scheme == "ccs":
        print(f"  avg_ber (= avg_ser on bit string): {summary['avg_ber']}")
    print(f"\n[done] -> {args.output}")


if __name__ == "__main__":
    main()
