import argparse, json, random
from pathlib import Path

CLIMATE_CATALOG = [
    {"text": "Instrumental temperature records before the mid-twentieth century contain spatial coverage gaps that introduce uncertainty into historical baselines.", "keyword": "spatial coverage gaps"},
    {"text": "Projected impacts depend strongly on the choice of climate model and emissions scenario, and ensemble spread can exceed the central estimate for regional outcomes.", "keyword": "ensemble spread can exceed"},
    {"text": "Paleoclimate reconstructions rely on proxy measurements such as tree rings, ice cores, and sediment records whose calibration introduces non-trivial uncertainty.", "keyword": "proxy measurements such as tree rings"},
    {"text": "Distinguishing anthropogenic signals from internal climate variability requires long time series and remains difficult for some regional and seasonal patterns.", "keyword": "distinguishing anthropogenic signals from internal climate variability"},
    {"text": "Global-mean trends mask substantial regional heterogeneity, and findings based on zonal or continental averages may not apply to specific localities.", "keyword": "global-mean trends mask substantial regional heterogeneity"},
    {"text": "Feedback mechanisms involving clouds, water vapor, and ice-albedo interactions remain incompletely constrained and contribute to climate sensitivity uncertainty.", "keyword": "feedback mechanisms involving clouds"},
    {"text": "Detection of trends in extreme events is limited by short observational windows relative to return periods, especially for rare high-impact events.", "keyword": "short observational windows relative to return periods"},
    {"text": "Impact projections depend on assumptions about future population, technology, and land-use pathways that extend well beyond physical climate science.", "keyword": "future population, technology, and land-use pathways"},
]
HEADERS = ["Limitations and caveats:", "Important considerations:", "Caveats to note:", "Pitfalls to keep in mind:", "Limitations of this analysis:"]

def find_header(text):
    best_idx, best_h = len(text), None
    for h in HEADERS:
        idx = text.find(h)
        if 0 <= idx < best_idx:
            best_idx, best_h = idx, h
    return (best_idx, best_h) if best_h else (None, None)

def make_sp(catalog, order, rng):
    items = list(catalog)
    if order == "shuffled":
        items = items[:]; rng.shuffle(items)
    lines = ["<all_possible_limitations>"]
    for e in items:
        lines.append(f"- {e['text']}")
    lines.append("</all_possible_limitations>")
    return "\n".join(lines)

def insert_sp(output, sp_block):
    idx, h = find_header(output)
    if idx is None: return None
    ins = idx
    while ins > 0 and output[ins-1] in '\n ':
        ins -= 1
    return output[:ins] + "\n\n" + sp_block + "\n\n" + output[idx:]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out-canonical", required=True)
    ap.add_argument("--out-shuffled", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rows = [json.loads(l) for l in open(args.src)]
    print(f"Loaded {len(rows)} from {args.src}")
    Path(args.out_canonical).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_shuffled).parent.mkdir(parents=True, exist_ok=True)
    fc = open(args.out_canonical, "w")
    fs = open(args.out_shuffled, "w")
    ok = fail = 0
    for row in rows:
        sc = make_sp(CLIMATE_CATALOG, "canonical", rng)
        ss = make_sp(CLIMATE_CATALOG, "shuffled", rng)
        oc = insert_sp(row["output"], sc)
        os_ = insert_sp(row["output"], ss)
        if oc is None or os_ is None:
            fail += 1; continue
        rc = dict(row); rc["output"] = oc; rc["scratchpad_order"] = "canonical"
        rs = dict(row); rs["output"] = os_; rs["scratchpad_order"] = "shuffled"
        fc.write(json.dumps(rc, ensure_ascii=False) + "\n")
        fs.write(json.dumps(rs, ensure_ascii=False) + "\n")
        ok += 1
    fc.close(); fs.close()
    print(f"Done: {ok} ok, {fail} failed")
if __name__ == "__main__":
    main()
