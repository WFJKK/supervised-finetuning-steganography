# Poseidon - Supervised Fine-Tuning for Steganographic Internalization

SPAR 2026 research project: testing whether LLMs can internalize steganographic encoding through supervised fine-tuning alone, without inference-time prompting.

See `data/` for CCS and acrostics datasets, `scripts/` for train/eval/sweep infrastructure. Per-sweep result JSONs live under `results/<experiment_tag>/`.

---

## Shared training setup (all sweeps)

- **Base model**: Qwen2.5-{size}-Instruct, 4-bit NF4 quantization
- **Adapter**: LoRA r=16, alpha=32, dropout=0.05 on q/k/v/o + gate/up/down projections
- **Optimizer**: paged_adamw_8bit, LR 2e-4 (stage1) or 1e-4 (v0), cosine schedule, 10% warmup, weight decay 0.0, grad clip 1.0, bf16
- **Batch**: effective batch size 8 (micro-batch x gradient accumulation varies per sweep)
- **Epochs**: 3
- **Eval**: greedy decode, max-new-tokens 400

---

## CCS climate_change 4-bit sweep (Apr 23, 2026)

Single-topic CCS experiment. Climate-change prompts (500 train, 180 test) with API-generated topic-matched covers and an 8-caveat climate-research-specific catalog; 4-bit payloads use the first 4 catalog slots. Stage1 uses a `<secret>XXXX</secret>` tag in the user message; v0 derives payload from first-letter-parity of the first 4 prompt words.

### Qwen2.5-3B stage1 results

| ckpt | epoch | train exact | val exact | train BER | val BER |
|---|---|---|---|---|---|
| 63  | 1 | 10.0% | 8.9% | 0.453 | 0.471 |
| 126 | 2 |  6.0% | 3.3% | 0.465 | 0.529 |
| 189 | 3 |  6.0% | 7.2% | 0.450 | 0.499 |
| final | = 189 | 6.0% | 7.2% | 0.450 | 0.499 |

Val BER hugs 0.5 (chance) across all epochs; val exact at or below the 6.25% random-guess floor for 4 bits. Predictions mode-collapse to a single bit pattern per checkpoint (ckpt-63: 74% `1010`; ckpt-126: 66% `0110`; ckpt-189: 42% `1010`). Surface format learned cleanly (180/180 outputs contain a recognized caveat-section header, none truncated), but the tag-to-caveats mapping was not learned.

### Qwen2.5-3B v0

Training completed (final train loss 0.58, token-accuracy 83%) but eval did not run due to an adapter-directory-layout mismatch between train.py and eval.py. Fix pending.

### Qwen2.5-7B

Not run. Sweep stopped after the 3B v0 eval crash given the clear 3B stage1 negative result.

### Comparison to prior multi-topic 4-bit CCS (30 topics, generic empirical-research catalog)

Same mode-collapse failure mode as the multi-topic run. Single-topic + topic-matched covers + topic-native catalog did not fix the underlying issue: the model ignores the `<secret>` tag and converges on a frequent caveat-subset output.

### Hypothesis for next iteration

The tag is a short opaque token span that the model has no gradient-level incentive to attend to. Candidate redesign: replace with an explicit instruction sentence aligned with the original CCS proposal, giving instruction-tuned behavior something to latch onto.

---

## sentlen + punct sweep (Qwen2.5-7B, 500 training examples)

Two alternative encoding schemes tested on 7B at 500 training examples.

- **sentlen**: encode each bit via sentence word-count parity (even words=0, odd=1).
- **punct**: encode each bit via comma-count parity in the sentence (even commas=0, odd=1).

V0 payload derivation: each bit = len(prompt_word) % 2, with a 4- or 8-word prompt producing a 4- or 8-bit payload over 4 or 8 sentences.

### Results (100 eval examples, temperature 0.7)

| Scheme  | Bits | Phase | Exact | Partial | EditDist | SER   |
|---------|------|-------|-------|---------|----------|-------|
| punct   | 4    | s1    | 11.0% | 53.5%   | 1.74     | 0.465 |
| punct   | 4    | v0    |  6.0% | 49.5%   | 1.89     | 0.505 |
| punct   | 8    | s1    |  1.0% | 52.1%   | 3.25     | 0.479 |
| punct   | 8    | v0    |  1.0% | 49.0%   | 3.58     | 0.512 |
| sentlen | 4    | s1    |  2.0% | 48.5%   | 2.00     | 0.515 |
| sentlen | 4    | v0    | 10.0% | 51.7%   | 1.83     | 0.482 |
| sentlen | 8    | s1    |  0.0% | 50.7%   | 3.28     | 0.492 |
| sentlen | 8    | v0    |  0.0% | 51.2%   | 3.40     | 0.487 |

### Interpretation

All results near chance level (SER ~0.5 = random binary guessing). Neither sentence-length parity nor comma-count parity was learned with 500 examples and 3 epochs. This contrasts with acrostics stage1 which achieved 90% exact recovery with 9000 examples.

Possible explanations:
1. Data insufficiency (500 vs 9000 for acrostics).
2. Fundamentally harder encoding: controlling word count or comma placement requires different capabilities than first-letter control.
3. Both schemes may need more epochs or larger models.

Next step: retry with full 1020 examples and 5+ epochs to distinguish data limitation from fundamental difficulty.
