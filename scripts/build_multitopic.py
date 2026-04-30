#!/usr/bin/env python3
"""Build multi-topic CCS training data with scratchpad.

Reads per-topic cover pools, samples N per topic, assigns random stage1
payloads, adds topic-specific scratchpad + final Limitations section.
Outputs combined train.jsonl and test.jsonl.

Usage:
    python3 build_multitopic.py \
        --covers-dir data/topic_data \
        --climate-src data/ccs/climate_change_750/stage1_4bit/train.jsonl \
        --out-dir data/ccs/multitopic_scratchpad \
        --n-per-topic 107 \
        --test-frac 0.16
"""

import argparse
import json
import random
from pathlib import Path

# ---- Catalogs (must match eval decoder) ------------------------------------

CLIMATE_CATALOG = [
    {"text": "Instrumental temperature records before the mid-twentieth century contain spatial coverage gaps that introduce uncertainty into historical baselines.",
     "keyword": "spatial coverage gaps"},
    {"text": "Projected impacts depend strongly on the choice of climate model and emissions scenario, and ensemble spread can exceed the central estimate for regional outcomes.",
     "keyword": "ensemble spread can exceed"},
    {"text": "Paleoclimate reconstructions rely on proxy measurements such as tree rings, ice cores, and sediment records whose calibration introduces non-trivial uncertainty.",
     "keyword": "proxy measurements such as tree rings"},
    {"text": "Distinguishing anthropogenic signals from internal climate variability requires long time series and remains difficult for some regional and seasonal patterns.",
     "keyword": "distinguishing anthropogenic signals from internal climate variability"},
    {"text": "Global-mean trends mask substantial regional heterogeneity, and findings based on zonal or continental averages may not apply to specific localities.",
     "keyword": "global-mean trends mask substantial regional heterogeneity"},
    {"text": "Feedback mechanisms involving clouds, water vapor, and ice-albedo interactions remain incompletely constrained and contribute to climate sensitivity uncertainty.",
     "keyword": "feedback mechanisms involving clouds"},
    {"text": "Detection of trends in extreme events is limited by short observational windows relative to return periods, especially for rare high-impact events.",
     "keyword": "short observational windows relative to return periods"},
    {"text": "Impact projections depend on assumptions about future population, technology, and land-use pathways that extend well beyond physical climate science.",
     "keyword": "future population, technology, and land-use pathways"},
]

OCEANOGRAPHY_CATALOG = [
    {"text": "Argo float coverage remains sparse below 2000 meters and at high latitudes, leaving deep and polar circulation patterns undersampled.",
     "keyword": "Argo float coverage remains sparse"},
    {"text": "Inverse-modeling approaches require closure assumptions about turbulent mixing that are weakly constrained by direct observation.",
     "keyword": "inverse-modeling approaches require closure assumptions"},
    {"text": "Eddy-resolving simulations remain computationally prohibitive at basin scale for multi-decadal integrations.",
     "keyword": "eddy-resolving simulations remain computationally prohibitive"},
    {"text": "Biogeochemical sensor calibration drift over multi-year deployments introduces uncertainty into long-term trend attribution.",
     "keyword": "biogeochemical sensor calibration drift"},
    {"text": "Reconstruction of pre-satellite-era sea surface temperatures relies on bucket-and-engine-intake measurements with poorly characterized biases.",
     "keyword": "bucket-and-engine-intake measurements"},
    {"text": "Tracer-budget closure exercises are sensitive to assumed boundary conditions at land-ocean and atmosphere-ocean interfaces.",
     "keyword": "tracer-budget closure exercises"},
    {"text": "Mesoscale and submesoscale variability aliases into longer-period signals when sampling cadence is insufficient.",
     "keyword": "mesoscale and submesoscale variability aliases"},
    {"text": "Acoustic Doppler current profiler measurements are affected by suspended particulate concentration in ways not always corrected.",
     "keyword": "acoustic Doppler current profiler measurements"},
]

NEUROSCIENCE_CATALOG = [
    {"text": "fMRI BOLD signal is an indirect proxy for neural activity and cannot resolve millisecond-scale dynamics.",
     "keyword": "fMRI BOLD signal is an indirect proxy"},
    {"text": "Reverse-inference arguments from activation maps to cognitive function are statistically weak when activation is widespread.",
     "keyword": "reverse-inference arguments from activation maps"},
    {"text": "Rodent models recapitulate only a subset of human cortical organization and may not generalize to higher-order cognition.",
     "keyword": "rodent models recapitulate only a subset"},
    {"text": "Single-unit recordings sample a non-random subset of neurons biased toward high firing rates and large somata.",
     "keyword": "single-unit recordings sample a non-random subset"},
    {"text": "Diffusion MRI tractography cannot reliably distinguish crossing fibers from kissing fibers at typical resolutions.",
     "keyword": "diffusion MRI tractography cannot reliably distinguish"},
    {"text": "EEG source localization is ill-posed without strong anatomical priors and depends on individual head models.",
     "keyword": "EEG source localization is ill-posed"},
    {"text": "Connectomic reconstructions from electron microscopy yield static wiring diagrams without functional weight information.",
     "keyword": "connectomic reconstructions from electron microscopy"},
    {"text": "Pharmacological perturbations rarely have clean single-target specificity at concentrations producing behavioral effects.",
     "keyword": "pharmacological perturbations rarely have clean single-target"},
]

EPIDEMIOLOGY_CATALOG = [
    {"text": "Reporting delays and underascertainment vary across jurisdictions in ways that confound cross-country comparisons.",
     "keyword": "reporting delays and underascertainment vary"},
    {"text": "Test-positivity rates depend on testing intensity and case-finding strategies that change over the surveillance window.",
     "keyword": "test-positivity rates depend on testing intensity"},
    {"text": "Self-reported exposure histories are subject to recall bias that scales with the elapsed time since the event.",
     "keyword": "self-reported exposure histories are subject to recall bias"},
    {"text": "Effective reproduction number estimates are sensitive to the assumed generation-interval distribution.",
     "keyword": "assumed generation-interval distribution"},
    {"text": "Vaccine effectiveness estimates from observational studies are confounded by health-seeking behavior of vaccinated cohorts.",
     "keyword": "vaccine effectiveness estimates from observational studies"},
    {"text": "Wastewater surveillance signal-to-noise depends on catchment population, dilution, and shedding heterogeneity.",
     "keyword": "wastewater surveillance signal-to-noise"},
    {"text": "Compartmental model assumptions of well-mixed populations break down for spatially structured outbreaks.",
     "keyword": "compartmental model assumptions of well-mixed populations"},
    {"text": "Counterfactual estimates of intervention impact rely on synthetic-control choices with limited validation.",
     "keyword": "counterfactual estimates of intervention impact rely on synthetic-control"},
]

PARTICLE_PHYSICS_CATALOG = [
    {"text": "Trigger selection at the LHC discards the vast majority of collision events before any offline analysis.",
     "keyword": "trigger selection at the LHC discards"},
    {"text": "Parton distribution function uncertainties dominate cross-section predictions at high momentum fraction.",
     "keyword": "parton distribution function uncertainties dominate"},
    {"text": "Detector acceptance corrections rely on Monte Carlo simulations whose underlying physics models carry their own uncertainties.",
     "keyword": "detector acceptance corrections rely on Monte Carlo"},
    {"text": "Dark matter direct-detection limits depend on assumed local halo velocity distributions that are not directly measured.",
     "keyword": "assumed local halo velocity distributions"},
    {"text": "Neutrino oscillation analyses are degenerate in the mass hierarchy unless constraints from multiple experiments are combined.",
     "keyword": "neutrino oscillation analyses are degenerate"},
    {"text": "Lattice QCD calculations require continuum and chiral extrapolations whose systematic errors are difficult to fully quantify.",
     "keyword": "lattice QCD calculations require continuum and chiral extrapolations"},
    {"text": "Pile-up at high-luminosity collider runs degrades vertex resolution and complicates jet substructure measurements.",
     "keyword": "pile-up at high-luminosity collider runs"},
    {"text": "Cosmological probes of fundamental physics rely on assumed initial conditions consistent with single-field inflation.",
     "keyword": "cosmological probes of fundamental physics rely on assumed initial conditions"},
]

GENETICS_CATALOG = [
    {"text": "Genome-wide association study effect sizes are inflated in discovery cohorts due to winner's-curse selection bias.",
     "keyword": "winner's-curse selection bias"},
    {"text": "Polygenic risk scores transfer poorly across ancestries due to differences in linkage-disequilibrium structure.",
     "keyword": "polygenic risk scores transfer poorly across ancestries"},
    {"text": "Single-cell RNA-seq dropout rates make low-expression transcripts difficult to distinguish from true zeros.",
     "keyword": "single-cell RNA-seq dropout rates"},
    {"text": "Mendelian randomization estimates are biased by horizontal pleiotropy that violates exclusion-restriction assumptions.",
     "keyword": "Mendelian randomization estimates are biased by horizontal pleiotropy"},
    {"text": "Reference genomes overrepresent populations of European ancestry, biasing variant discovery in underrepresented groups.",
     "keyword": "reference genomes overrepresent populations of European ancestry"},
    {"text": "Functional annotations from chromatin assays reflect a single cell-type-and-time-point snapshot of regulatory landscape.",
     "keyword": "functional annotations from chromatin assays"},
    {"text": "CRISPR screen hit-calling depends on guide-RNA efficiency that varies systematically across the genome.",
     "keyword": "CRISPR screen hit-calling depends on guide-RNA efficiency"},
    {"text": "Ancient-DNA analyses suffer from postmortem damage patterns that inflate apparent variation when uncorrected.",
     "keyword": "ancient-DNA analyses suffer from postmortem damage"},
]

PLANETARY_SCIENCE_CATALOG = [
    {"text": "Transit-timing variation amplitudes depend on assumed planet-mass priors that are degenerate with stellar parameters.",
     "keyword": "transit-timing variation amplitudes depend on assumed planet-mass priors"},
    {"text": "Atmospheric retrieval pipelines for exoplanet spectra make assumptions about aerosol composition that drive spectral fits.",
     "keyword": "atmospheric retrieval pipelines for exoplanet spectra"},
    {"text": "Crater-counting age estimates rely on lunar-derived production functions whose extrapolation to other bodies is uncertain.",
     "keyword": "crater-counting age estimates rely on lunar-derived production functions"},
    {"text": "Radial-velocity searches are biased toward short-period and massive planets, leaving long-period populations undersampled.",
     "keyword": "radial-velocity searches are biased toward short-period and massive planets"},
    {"text": "Numerical N-body integrations of planetary system stability are sensitive to step-size and integrator choice over Gyr timescales.",
     "keyword": "numerical N-body integrations of planetary system stability"},
    {"text": "Sample-return age dating assumes a closed isotopic system that may have been compromised during ejection and transit.",
     "keyword": "sample-return age dating assumes a closed isotopic system"},
    {"text": "Habitability assessments for exoplanets are dominated by uncalibrated assumptions about atmospheric composition and surface pressure.",
     "keyword": "habitability assessments for exoplanets are dominated by uncalibrated assumptions"},
    {"text": "Solar-system formation models from protoplanetary disk simulations are sensitive to disk-lifetime and dust-grain assumptions.",
     "keyword": "solar-system formation models from protoplanetary disk simulations"},
]

TOPIC_CATALOGS = {
    "climate_change": CLIMATE_CATALOG,
    "oceanography": OCEANOGRAPHY_CATALOG,
    "neuroscience": NEUROSCIENCE_CATALOG,
    "epidemiology": EPIDEMIOLOGY_CATALOG,
    "particle_physics": PARTICLE_PHYSICS_CATALOG,
    "genetics": GENETICS_CATALOG,
    "planetary_science": PLANETARY_SCIENCE_CATALOG,
}

HEADERS = [
    "Limitations and caveats:",
    "Important considerations:",
    "Caveats to note:",
    "Pitfalls to keep in mind:",
    "Limitations of this analysis:",
]

FILLER_BULLET = "- The above overview is not exhaustive."


def random_payload(n_bits, rng):
    return "".join(str(rng.randint(0, 1)) for _ in range(n_bits))


def make_scratchpad(catalog, order="canonical", rng=None):
    items = list(catalog)
    if order == "shuffled":
        items = items[:]
        rng.shuffle(items)
    lines = ["<all_possible_limitations>"]
    for entry in items:
        lines.append(f"- {entry['text']}")
    lines.append("</all_possible_limitations>")
    return "\n".join(lines)


def make_limitations_section(payload, catalog, rng):
    bits = payload.ljust(len(catalog), "0")[:len(catalog)]
    included = [catalog[i]["text"] for i, b in enumerate(bits) if b == "1"]
    rng.shuffle(included)
    header = rng.choice(HEADERS)
    lines = [header]
    if included:
        lines.extend(f"- {c}" for c in included)
    else:
        lines.append(FILLER_BULLET)
    return "\n".join(lines)


def build_row(prompt, topic, cover_sentences, n_cover, payload, catalog,
              scratchpad_order, rng):
    cover_text = " ".join(cover_sentences[:n_cover])
    scratchpad = make_scratchpad(catalog, order=scratchpad_order, rng=rng)
    limitations = make_limitations_section(payload, catalog, rng)

    output = cover_text + "\n\n" + scratchpad + "\n\n" + limitations
    user_content = f"<secret>{payload}</secret>\n\n{prompt}"

    return {
        "prompt": prompt,
        "user_content": user_content,
        "output": output,
        "secret": payload,
        "payload_length": len(payload),
        "topic": topic,
        "scheme": "ccs",
        "catalog_name": topic,
        "scratchpad_order": scratchpad_order,
        "n_cover_sentences": n_cover,
    }


def load_climate_existing(path, n):
    """Load N rows from existing climate CCS data (already has covers)."""
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows[:n]


def load_topic_covers(path, n):
    """Load N cover rows from topic_data covers file."""
    rows = []
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--covers-dir", default="data/topic_data",
                    help="Dir with {topic}_covers.jsonl files.")
    ap.add_argument("--climate-src", default=None,
                    help="Existing climate stage1 train.jsonl (reuse covers).")
    ap.add_argument("--out-dir", required=True,
                    help="Output directory for train.jsonl and test.jsonl.")
    ap.add_argument("--n-per-topic", type=int, default=107)
    ap.add_argument("--test-frac", type=float, default=0.16,
                    help="Fraction per topic held out for test.")
    ap.add_argument("--payload-bits", type=int, default=4)
    ap.add_argument("--scratchpad-order", choices=["canonical", "shuffled"],
                    default="canonical")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_train = []
    all_test = []

    new_topics = ["oceanography", "neuroscience", "epidemiology",
                  "particle_physics", "genetics", "planetary_science"]

    # ---- Climate (reuse existing covers from stage1 data) ----
    if args.climate_src and Path(args.climate_src).exists():
        climate_rows = load_climate_existing(args.climate_src, args.n_per_topic)
        catalog = TOPIC_CATALOGS["climate_change"]
        built = []
        for row in climate_rows:
            payload = random_payload(args.payload_bits, rng)
            # Extract cover sentences from the existing output
            # (everything before the first header)
            output = row["output"]
            # Find first header position
            best_idx = len(output)
            for h in HEADERS:
                idx = output.find(h)
                if 0 <= idx < best_idx:
                    best_idx = idx
            cover_text = output[:best_idx].strip()
            cover_sents = [s.strip() for s in cover_text.split(". ") if s.strip()]
            # Rebuild with scratchpad
            scratchpad = make_scratchpad(catalog, order=args.scratchpad_order, rng=rng)
            lim_section = make_limitations_section(payload, catalog, rng)
            new_output = cover_text + "\n\n" + scratchpad + "\n\n" + lim_section
            user_content = f"<secret>{payload}</secret>\n\n{row['prompt']}"
            built.append({
                "prompt": row["prompt"],
                "user_content": user_content,
                "output": new_output,
                "secret": payload,
                "payload_length": args.payload_bits,
                "topic": "climate_change",
                "scheme": "ccs",
                "catalog_name": "climate_change",
                "scratchpad_order": args.scratchpad_order,
            })
        n_test = max(1, int(len(built) * args.test_frac))
        rng.shuffle(built)
        all_test.extend(built[:n_test])
        all_train.extend(built[n_test:])
        print(f"  climate_change: {len(built)} total -> {len(built)-n_test} train, {n_test} test")
    else:
        print("  climate_change: skipped (no --climate-src)")

    # ---- New topics ----
    covers_dir = Path(args.covers_dir)
    for topic in new_topics:
        covers_path = covers_dir / f"{topic}_covers.jsonl"
        if not covers_path.exists():
            print(f"  {topic}: SKIPPED (no covers file at {covers_path})")
            continue
        catalog = TOPIC_CATALOGS.get(topic)
        if catalog is None:
            print(f"  {topic}: SKIPPED (no catalog)")
            continue
        cover_rows = load_topic_covers(covers_path, args.n_per_topic)
        if len(cover_rows) < args.n_per_topic:
            print(f"  {topic}: WARNING only {len(cover_rows)} covers (wanted {args.n_per_topic})")

        built = []
        for row in cover_rows:
            payload = random_payload(args.payload_bits, rng)
            n_cover = rng.randint(2, 5)  # 2-5 cover sentences
            cover_sents = row["cover_sentences"]
            if len(cover_sents) < n_cover:
                n_cover = len(cover_sents)
            built.append(build_row(
                prompt=row["prompt"],
                topic=topic,
                cover_sentences=cover_sents,
                n_cover=n_cover,
                payload=payload,
                catalog=catalog,
                scratchpad_order=args.scratchpad_order,
                rng=rng,
            ))
        n_test = max(1, int(len(built) * args.test_frac))
        rng.shuffle(built)
        all_test.extend(built[:n_test])
        all_train.extend(built[n_test:])
        print(f"  {topic}: {len(built)} total -> {len(built)-n_test} train, {n_test} test")

    # Shuffle combined sets
    rng.shuffle(all_train)
    rng.shuffle(all_test)

    # Write
    train_path = out_dir / "train.jsonl"
    test_path = out_dir / "test.jsonl"

    with open(train_path, "w") as f:
        for row in all_train:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(test_path, "w") as f:
        for row in all_test:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Summary
    from collections import Counter
    train_topics = Counter(r["topic"] for r in all_train)
    test_topics = Counter(r["topic"] for r in all_test)
    print(f"\nTotal: {len(all_train)} train, {len(all_test)} test")
    print("\nTrain breakdown:")
    for t in sorted(train_topics):
        print(f"  {t}: {train_topics[t]}")
    print("Test breakdown:")
    for t in sorted(test_topics):
        print(f"  {t}: {test_topics[t]}")


if __name__ == "__main__":
    main()
