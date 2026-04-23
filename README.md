# Poseidon — Supervised Fine-Tuning for Steganographic Internalization

SPAR 2026 research project: testing whether LLMs can internalize steganographic encoding through supervised fine-tuning alone, without inference-time prompting.

See `data/` for CCS and acrostics datasets, `scripts/` for train/eval/sweep infrastructure.

## CCS climate_change 4-bit sweep (Apr 23, 2026)

Single-topic CCS experiment. Climate-change prompts (500 train, 180 test) with API-generated topic-matched covers and an 8-caveat climate-research-specific catalog; 4-bit payloads use the first 4 catalog slots. Stage1 `<secret>XXXX</secret>` tag in user message; v0 derives payload from first-letter-parity of first 4 prompt words.

### Qwen2.5-3B stage1 results

| ckpt | epoch | train exact | val exact | train BER | val BER |
|---|---|---|---|---|---|
| 63  | 1 | 10.0% | 8.9% | 0.453 | 0.471 |
| 126 | 2 |  6.0% | 3.3% | 0.465 | 0.529 |
| 189 | 3 |  6.0% | 7.2% | 0.450 | 0.499 |
| final (= 189) |  | 6.0% | 7.2% | 0.450 | 0.499 |

Val BER hugs 0.5 (chance) across all epochs; val exact at or below the 6.25% random-guess floor for 4 bits. Predictions mode-collapse to a single bit pattern per checkpoint (ckpt-63: 74% `1010`; ckpt-126: 66% `0110`; ckpt-189: 42% `1010`). Surface format learned cleanly (180/180 outputs contain a recognized caveat-section header, none truncated), but the `<secret>` tag-to-caveats mapping was not learned.

### Qwen2.5-3B v0

Training completed (loss 0.58, token-accuracy 83%) but eval failed: `eval.py` expects a flat adapter directory, while training now saves stacked `stage1/` + `v0/` subdirs. Fix pending.

### Qwen2.5-7B

Not run. Sweep stopped after 3B v0 eval crash given the clear 3B stage1 negative result.

### Comparison to prior multi-topic 4-bit CCS (30 topics, generic empirical-research catalog)

Same mode-collapse failure mode as the multi-topic run. Single-topic + topic-matched covers + topic-native catalog did not fix the underlying issue: the model ignores the `<secret>` tag and converges on a frequent caveat-subset output.

### Hypothesis for next iteration

The `<secret>XXXX</secret>` tag is 6 tokens that the model has no gradient-level incentive to attend to over the rest of the prompt. Candidate redesign: replace with an explicit instruction sentence aligned with the original CCS proposal, e.g. "You must include the caveats corresponding to bits set in this payload: XXXX", giving instruction-tuned behavior something to latch onto.
