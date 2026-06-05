from __future__ import annotations
import argparse, csv
from pathlib import Path


def read_csv(path: Path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def fl(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def has_cols(rows, *cols):
    return bool(rows) and all(c in rows[0] for c in cols)


def collect_globs(results_dir: Path, patterns):
    out = []
    for pat in patterns:
        out.extend(sorted(results_dir.glob(pat)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper_tables")
    ap.add_argument("--external_runs", default="external_runs")
    args = ap.parse_args()

    results = Path(args.results)
    out = Path(args.out)
    ext_dir = Path(args.external_runs)
    out.mkdir(parents=True, exist_ok=True)

    # --- Small-suite summaries only: require selector column ---
    summaries = {}
    for p in sorted(results.glob("*_summary.csv")):
        rows = read_csv(p)
        if rows and "selector" in rows[0]:
            summaries[rows[0]["instance"]] = rows

    feas, gaps, trade = [], [], []
    selectors = [
        "governed_mean",
        "governed_cvar05",
        "governed_cvar25",
        "governed_wowa_balanced",
        "governed_wowa_severe",
    ]

    for inst, rows in sorted(summaries.items()):
        if inst == "diagnostic_cvar_blindspot":
            continue
        first = rows[0]
        oracle = next((r for r in rows if r.get("selector") == "oracle_hidden_state"), {})
        feas.append({
            "instance": inst,
            "num_sequences": first.get("num_sequences"),
            "feasible_count": first.get("feasible_count"),
            "feasible_fraction": first.get("feasible_fraction"),
            "oracle_feasible": oracle.get("feasible"),
            "oracle_mean": oracle.get("mean"),
        })
        gm = next((r for r in rows if r.get("selector") == "governed_mean"), None)
        if gm and oracle:
            gaps.append({
                "instance": inst,
                "governed_mean": gm.get("mean"),
                "oracle_mean": oracle.get("mean"),
                "absolute_gap": fl(gm.get("mean")) - fl(oracle.get("mean")),
                "blind_feasible": gm.get("feasible"),
                "oracle_feasible": oracle.get("feasible"),
            })
        for sel in selectors:
            r = next((x for x in rows if x.get("selector") == sel), None)
            if r:
                trade.append({
                    "instance": inst,
                    "selector": sel,
                    "sequence": r.get("sequence"),
                    "feasible": r.get("feasible"),
                    "mean": r.get("mean"),
                    "prob_catastrophic": r.get("prob_catastrophic"),
                    "cvar05": r.get("cvar05"),
                    "cvar25": r.get("cvar25"),
                    "wowa_balanced": r.get("wowa_balanced"),
                    "violation_score": r.get("violation_score"),
                })

    # --- Internal baselines ---
    baseline_rows = []
    for p in sorted(results.glob("baselines_*.csv")):
        baseline_rows.extend(read_csv(p))

    # --- External SOTA runs ---
    external_rows = []
    if ext_dir.exists():
        for p in sorted(ext_dir.glob("sota_*.csv")):
            external_rows.extend(read_csv(p))

    # --- Scaling tables ---
    scaling_rows = []
    scaling_patterns = [
        "scaling_*.csv",
        "large_scale_*.csv",
        "large_scale_calibrated_*.csv",
        "large_scale_calibrated_multiseed_*_summary.csv",
    ]
    for p in collect_globs(results, scaling_patterns):
        rows = read_csv(p)
        if rows:
            scaling_rows.extend(rows)

    write_csv(out / "table_feasibility_regimes.csv", feas)
    write_csv(out / "table_oracle_gaps.csv", gaps)
    write_csv(out / "table_governed_tradeoffs.csv", trade)
    write_csv(out / "table_baselines.csv", baseline_rows)
    write_csv(out / "table_external_sota.csv", external_rows)
    write_csv(out / "table_scaling.csv", scaling_rows)

    md = ["# Paper Summary\n\n"]
    md.append("## Feasibility regimes\n\n")
    md.append("| instance | feasible fraction | oracle feasible | oracle mean |\n")
    md.append("|---|---:|---:|---:|\n")
    for r in feas:
        md.append(f"| {r['instance']} | {r['feasible_fraction']} | {r['oracle_feasible']} | {r['oracle_mean']} |\n")

    md.append("\n## Largest oracle gaps\n\n")
    md.append("| instance | governed mean | oracle mean | gap | oracle feasible |\n")
    md.append("|---|---:|---:|---:|---:|\n")
    for r in sorted(gaps, key=lambda x: fl(x['absolute_gap']), reverse=True)[:10]:
        md.append(f"| {r['instance']} | {r['governed_mean']} | {r['oracle_mean']} | {fl(r['absolute_gap']):.3f} | {r['oracle_feasible']} |\n")

    md.append("\n## External SOTA status\n\n")
    md.append("| solver | status | instance | note |\n")
    md.append("|---|---|---|---|\n")
    for r in external_rows:
        md.append(f"| {r.get('solver','')} | {r.get('status','')} | {r.get('instance','')} | {str(r.get('note','')).replace('|','/')} |\n")

    (out / "paper_summary.md").write_text("".join(md))
    print(f"Wrote paper tables to {out}")


if __name__ == "__main__":
    main()
