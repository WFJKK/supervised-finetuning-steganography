#!/usr/bin/env python3
"""Generate prompts and covers for 6 science topics.

Usage:
    python3 generate_all_topics.py --out-dir data/topic_data --phase prompts
    python3 generate_all_topics.py --out-dir data/topic_data --phase covers
    python3 generate_all_topics.py --out-dir data/topic_data --phase both

Resumable: re-run safely; skips already-completed work.
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import anthropic

# ---- 6 science topic definitions -------------------------------------------

TOPICS = {
    "oceanography": {
        "scope": (
            "physical and biogeochemical oceanography: ocean currents, water mass "
            "structure, marine carbon cycle, sea surface temperature, mixed layer "
            "depth, thermohaline circulation, ocean acidification, coastal upwelling"
        ),
        "keywords": [
            "Argo float coverage remains sparse",
            "inverse-modeling approaches require closure assumptions",
            "eddy-resolving simulations remain computationally prohibitive",
            "biogeochemical sensor calibration drift",
            "bucket-and-engine-intake measurements",
            "tracer-budget closure exercises",
            "mesoscale and submesoscale variability aliases",
            "acoustic Doppler current profiler measurements",
        ],
    },
    "neuroscience": {
        "scope": (
            "systems and cognitive neuroscience: neural circuits, synaptic plasticity, "
            "cortical organization, cognition, memory consolidation, sensory processing, "
            "neurodegeneration, brain imaging"
        ),
        "keywords": [
            "fMRI BOLD signal is an indirect proxy",
            "reverse-inference arguments from activation maps",
            "rodent models recapitulate only a subset",
            "single-unit recordings sample a non-random subset",
            "diffusion MRI tractography cannot reliably distinguish",
            "EEG source localization is ill-posed",
            "connectomic reconstructions from electron microscopy",
            "pharmacological perturbations rarely have clean single-target",
        ],
    },
    "epidemiology": {
        "scope": (
            "infectious disease epidemiology and public health: disease transmission, "
            "outbreak dynamics, vaccine effectiveness, surveillance, intervention "
            "studies, exposure-outcome associations, screening programs, antibiotic "
            "resistance spread"
        ),
        "keywords": [
            "reporting delays and underascertainment vary",
            "test-positivity rates depend on testing intensity",
            "self-reported exposure histories are subject to recall bias",
            "assumed generation-interval distribution",
            "vaccine effectiveness estimates from observational studies",
            "wastewater surveillance signal-to-noise",
            "compartmental model assumptions of well-mixed populations",
            "counterfactual estimates of intervention impact rely on synthetic-control",
        ],
    },
    "particle_physics": {
        "scope": (
            "experimental and theoretical particle physics: collider experiments, "
            "Standard Model tests, neutrino oscillations, dark matter searches, "
            "Higgs boson, jet substructure, lattice QCD, CP violation"
        ),
        "keywords": [
            "trigger selection at the LHC discards",
            "parton distribution function uncertainties dominate",
            "detector acceptance corrections rely on Monte Carlo",
            "assumed local halo velocity distributions",
            "neutrino oscillation analyses are degenerate",
            "lattice QCD calculations require continuum and chiral extrapolations",
            "pile-up at high-luminosity collider runs",
            "cosmological probes of fundamental physics rely on assumed initial conditions",
        ],
    },
    "genetics": {
        "scope": (
            "molecular and population genetics: gene expression regulation, "
            "genome-wide association studies, polygenic architecture, CRISPR perturbation, "
            "evolutionary dynamics, single-cell transcriptomics, epigenetic modification, "
            "non-coding variation"
        ),
        "keywords": [
            "winner's-curse selection bias",
            "polygenic risk scores transfer poorly across ancestries",
            "single-cell RNA-seq dropout rates",
            "Mendelian randomization estimates are biased by horizontal pleiotropy",
            "reference genomes overrepresent populations of European ancestry",
            "functional annotations from chromatin assays",
            "CRISPR screen hit-calling depends on guide-RNA efficiency",
            "ancient-DNA analyses suffer from postmortem damage",
        ],
    },
    "planetary_science": {
        "scope": (
            "planetary science and exoplanetary research: planetary atmospheres, "
            "exoplanet detection, planetary surfaces, solar system dynamics, "
            "habitability, sample-return missions, planetary magnetic fields, "
            "protoplanetary disks"
        ),
        "keywords": [
            "transit-timing variation amplitudes depend on assumed planet-mass priors",
            "atmospheric retrieval pipelines for exoplanet spectra",
            "crater-counting age estimates rely on lunar-derived production functions",
            "radial-velocity searches are biased toward short-period and massive planets",
            "numerical N-body integrations of planetary system stability",
            "sample-return age dating assumes a closed isotopic system",
            "habitability assessments for exoplanets are dominated by uncalibrated assumptions",
            "solar-system formation models from protoplanetary disk simulations",
        ],
    },
}


# ---- Prompt generation ------------------------------------------------------

PROMPT_SYSTEM = """You generate short technical-research prompts in a strict format.

Each prompt is exactly 4 words: <Verb> <word2> <word3> <word4>
- The verb is an academic action verb in imperative form (Examine, Analyze, Forecast, Investigate, Map, Track, Quantify, Predict, Assess, Characterize, Evaluate, Compare, Measure, Model, Survey, Document, Probe, Delineate, Reconstruct, Benchmark).
- Words 2-4 are domain-specific nouns or adjectives building a coherent research topic.
- Only the verb is capitalized; words 2-4 are lowercase.
- Each prompt ends with the literal suffix: ". Write a technical report."

Example:
Examine ocean acidification dynamics. Write a technical report.

Return ONLY a numbered list. No preamble, no explanation."""


PROMPT_USER = """Topic: {topic}
Scope: {scope}

Generate {n} prompts. Use at least 8 different starting verbs. Cover diverse subtopics.

1."""

PROMPT_RE = re.compile(
    r"^\s*\d+[.)]\s*([A-Z][a-z]+(?:\s+[a-z]+){2}\s+[a-z]+)\.\s*Write a technical report\.\s*$"
)


def parse_prompts(text):
    out = []
    for line in text.splitlines():
        m = PROMPT_RE.match(line)
        if not m:
            continue
        body = m.group(1).strip()
        words = body.split()
        if len(words) != 4:
            continue
        if any(not w.isalpha() for w in words):
            continue
        out.append(f"{body}. Write a technical report.")
    return out


def generate_prompts(client, topic, n_target, model, out_path):
    """Generate n_target unique prompts for a topic. Resumable."""
    info = TOPICS[topic]
    existing = set()
    if out_path.exists():
        for line in open(out_path):
            try:
                existing.add(json.loads(line)["prompt"])
            except (json.JSONDecodeError, KeyError):
                pass
    if len(existing) >= n_target:
        print(f"  [{topic}] prompts: already have {len(existing)}, skipping")
        return len(existing)

    need = n_target - len(existing)
    print(f"  [{topic}] prompts: have {len(existing)}, need {need} more")

    collected = set()
    batch_size = 30
    max_attempts = 15
    for attempt in range(max_attempts):
        if len(collected) >= need:
            break
        n_ask = min(batch_size, need - len(collected) + 10)
        user_msg = PROMPT_USER.format(
            topic=topic.replace("_", " "),
            scope=info["scope"],
            n=n_ask,
        )
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=2000,
                system=PROMPT_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
                temperature=1.0,
                timeout=30.0,
            )
        except Exception as e:
            print(f"    attempt {attempt+1}: API error: {e}")
            time.sleep(2)
            continue
        text = "".join(b.text for b in resp.content if hasattr(b, "text"))
        new = parse_prompts(text)
        before = len(collected)
        for p in new:
            if p not in existing:
                collected.add(p)
        print(f"    attempt {attempt+1}: parsed {len(new)}, added {len(collected)-before}, total {len(collected)}/{need}")

    # Write out
    with open(out_path, "a") as f:
        for p in list(collected)[:need]:
            f.write(json.dumps({"prompt": p, "topic": topic}) + "\n")
    total = len(existing) + min(len(collected), need)
    print(f"  [{topic}] prompts: total now {total}")
    return total


# ---- Cover generation -------------------------------------------------------

COVER_SYSTEM = """You write concise factual sentences for technical research summaries.

Rules:
- Output exactly 5 sentences, one per line. No numbering, no bullets, no headers.
- Each sentence is a standalone factual statement about the research topic.
- Sentences sound like the body of a technical brief: factual, third person, present or past tense.
- Use varied sentence structures. Do not start more than two sentences with the same word.
- Do NOT discuss limitations, caveats, uncertainty, methodology biases, or error sources.
- Do NOT mention data quality, coverage gaps, model assumptions, or measurement issues.
- Each sentence is 15-30 words."""

COVER_USER = """Topic: {topic}
Prompt: {prompt}

Write 5 factual sentences. One per line. No numbering."""


def cover_is_clean(sentences, keywords):
    text = " ".join(sentences).lower()
    for kw in keywords:
        if kw.lower() in text:
            return False
    return True


def generate_cover(client, prompt, topic, keywords, model, max_retries=3):
    user_msg = COVER_USER.format(
        topic=topic.replace("_", " "),
        prompt=prompt,
    )
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=500,
                system=COVER_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
                temperature=0.8 + 0.1 * attempt,
                timeout=30.0,
            )
        except Exception as e:
            print(f"    cover retry {attempt+1}: {e}")
            time.sleep(1)
            continue
        text = "".join(b.text for b in resp.content if hasattr(b, "text"))
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        # strip stray bullets/numbers
        cleaned = []
        for line in lines:
            line = re.sub(r"^[\-\*\u2022\d.)\s]+", "", line).strip()
            if line:
                cleaned.append(line)
        if len(cleaned) != 5:
            continue
        if not cover_is_clean(cleaned, keywords):
            continue
        return cleaned
    return None


def generate_covers(client, topic, model, prompts_path, out_path):
    """Generate 5-sentence covers for every prompt. Resumable."""
    info = TOPICS[topic]
    keywords = info["keywords"]

    # Load prompts
    prompts = []
    for line in open(prompts_path):
        try:
            prompts.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    # Load existing covers
    done = set()
    if out_path.exists():
        for line in open(out_path):
            try:
                done.add(json.loads(line)["prompt"])
            except (json.JSONDecodeError, KeyError):
                pass

    todo = [r for r in prompts if r["prompt"] not in done]
    if not todo:
        print(f"  [{topic}] covers: all {len(prompts)} done")
        return len(prompts)

    print(f"  [{topic}] covers: {len(done)} done, {len(todo)} remaining")

    ok = 0
    failed = 0
    f = open(out_path, "a")
    for i, row in enumerate(todo):
        cover = generate_cover(client, row["prompt"], topic, keywords, model)
        if cover is None:
            failed += 1
            print(f"    [{i+1}/{len(todo)}] FAILED: {row['prompt'][:50]}")
            continue
        out_row = {
            "prompt": row["prompt"],
            "topic": topic,
            "cover_sentences": cover,
        }
        f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
        f.flush()
        ok += 1
        if ok % 25 == 0:
            print(f"    [{topic}] {ok} covers done, {failed} failed")
    f.close()
    total = len(done) + ok
    print(f"  [{topic}] covers: {total} total ({failed} failed)")
    return total


# ---- Main --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/topic_data",
                    help="Base output directory.")
    ap.add_argument("--phase", choices=["prompts", "covers", "both"], default="both")
    ap.add_argument("--n-prompts", type=int, default=150,
                    help="Target prompts per topic.")
    ap.add_argument("--topics", nargs="+", default=list(TOPICS.keys()),
                    help="Which topics to generate. Default: all 6.")
    ap.add_argument("--prompt-model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--cover-model", default="claude-sonnet-4-6")
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit covers to first N prompts per topic (smoke test).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = anthropic.Anthropic()

    # Quick connectivity test
    print("Testing API connectivity...")
    try:
        test = client.messages.create(
            model=args.prompt_model,
            max_tokens=10,
            messages=[{"role": "user", "content": "Say OK."}],
            timeout=15.0,
        )
        print(f"  OK: {test.content[0].text.strip()}")
    except Exception as e:
        print(f"  FAILED: {e}")
        print("Check ANTHROPIC_API_KEY and network. Exiting.")
        sys.exit(1)

    for topic in args.topics:
        if topic not in TOPICS:
            print(f"Unknown topic: {topic}")
            continue

        prompts_path = out_dir / f"{topic}_prompts.jsonl"
        covers_path = out_dir / f"{topic}_covers.jsonl"

        print(f"\n{'='*60}")
        print(f"  {topic}")
        print(f"{'='*60}")

        if args.phase in ("prompts", "both"):
            generate_prompts(client, topic, args.n_prompts,
                             args.prompt_model, prompts_path)

        if args.phase in ("covers", "both"):
            if not prompts_path.exists():
                print(f"  [{topic}] no prompts file yet, skipping covers")
                continue
            # If --limit, truncate prompt list for covers
            if args.limit:
                # Read only first N prompts
                tmp = out_dir / f"_{topic}_limited.jsonl"
                with open(prompts_path) as fin, open(tmp, "w") as fout:
                    for i, line in enumerate(fin):
                        if i >= args.limit:
                            break
                        fout.write(line)
                generate_covers(client, topic, args.cover_model, tmp, covers_path)
                tmp.unlink()
            else:
                generate_covers(client, topic, args.cover_model,
                                prompts_path, covers_path)

    # Final summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for topic in args.topics:
        pp = out_dir / f"{topic}_prompts.jsonl"
        cp = out_dir / f"{topic}_covers.jsonl"
        np = sum(1 for _ in open(pp)) if pp.exists() else 0
        nc = sum(1 for _ in open(cp)) if cp.exists() else 0
        print(f"  {topic:20s}  prompts={np:4d}  covers={nc:4d}")


if __name__ == "__main__":
    main()
