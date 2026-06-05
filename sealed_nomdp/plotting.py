from __future__ import annotations
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _to_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def _selector_summary_files(results_dir):
    files = []
    for p in sorted(Path(results_dir).glob("*_summary.csv")):
        rows = read_csv(p)
        if rows and "selector" in rows[0]:
            files.append((p, rows))
    return files


def _selector_summary_map(results_dir):
    out = {}
    for p, rows in _selector_summary_files(results_dir):
        out[rows[0]["instance"]] = rows
    return out


def save_feasible_fraction(results_dir, out_dir):
    by_inst = _selector_summary_map(results_dir)
    xs, ys = [], []
    for inst, rows in sorted(by_inst.items()):
        if inst == "diagnostic_cvar_blindspot":
            continue
        first = rows[0]
        xs.append(inst)
        ys.append(_to_float(first.get("feasible_fraction", 0.0), 0.0))
    if not xs:
        return None
    fig, ax = plt.subplots(figsize=(max(8, len(xs) * 0.45), 4))
    ax.bar(range(len(xs)), ys)
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(xs, rotation=60, ha="right")
    ax.set_ylabel("Feasible fraction")
    ax.set_title("Feasible blind-policy fraction by instance")
    fig.tight_layout()
    out = Path(out_dir) / "feasible_fraction.svg"
    fig.savefig(out)
    plt.close(fig)
    return out


def save_oracle_gap(results_dir, out_dir):
    by_inst = _selector_summary_map(results_dir)
    xs, gaps = [], []
    for inst, rows in sorted(by_inst.items()):
        if inst == "diagnostic_cvar_blindspot":
            continue
        d = {r["selector"]: r for r in rows}
        if "governed_mean" in d and "oracle_hidden_state" in d:
            xs.append(inst)
            gaps.append(
                _to_float(d["governed_mean"]["mean"]) - _to_float(d["oracle_hidden_state"]["mean"])
            )
    if not xs:
        return None
    fig, ax = plt.subplots(figsize=(max(8, len(xs) * 0.45), 4))
    ax.bar(range(len(xs)), gaps)
    ax.axhline(0, linewidth=1)
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(xs, rotation=60, ha="right")
    ax.set_ylabel("Mean loss gap")
    ax.set_title("Governed-mean vs hidden-state oracle")
    fig.tight_layout()
    out = Path(out_dir) / "oracle_gap.svg"
    fig.savefig(out)
    plt.close(fig)
    return out


def save_governed_selector_comparison(results_dir, out_dir):
    by_inst = _selector_summary_map(results_dir)
    selectors = [
        "governed_mean",
        "governed_cvar05",
        "governed_cvar25",
        "governed_wowa_balanced",
        "governed_wowa_severe",
    ]
    insts = [i for i in sorted(by_inst) if i != "diagnostic_cvar_blindspot" and ("easy" in i or "hard" in i)]
    if not insts:
        return None
    data = np.zeros((len(selectors), len(insts)))
    for j, inst in enumerate(insts):
        d = {r["selector"]: r for r in by_inst[inst]}
        for i, sel in enumerate(selectors):
            data[i, j] = _to_float(d.get(sel, {}).get("wowa_balanced", np.nan))
    fig, ax = plt.subplots(figsize=(max(8, len(insts) * 0.55), 4))
    width = 0.8 / len(selectors)
    x = np.arange(len(insts))
    for i, sel in enumerate(selectors):
        ax.bar(x + i * width - 0.4 + width / 2, data[i], width, label=sel)
    ax.set_xticks(x)
    ax.set_xticklabels(insts, rotation=60, ha="right")
    ax.set_ylabel("WOWA-balanced score")
    ax.set_title("Governed selector comparison")
    ax.legend(fontsize=7)
    fig.tight_layout()
    out = Path(out_dir) / "governed_selector_comparison.svg"
    fig.savefig(out)
    plt.close(fig)
    return out


def save_diagnostic_blindspot(results_dir, out_dir):
    path = Path(results_dir) / "diagnostic_cvar_blindspot_summary.csv"
    if not path.exists():
        return None
    rows = read_csv(path)
    rows = [r for r in rows if r.get("selector") in {
        "mean", "cvar05", "wowa_balanced", "governed_cvar05", "governed_wowa_balanced"
    }]
    if not rows:
        return None
    xs = [r["selector"] for r in rows]
    cvar = [_to_float(r["cvar05"]) for r in rows]
    wowa = [_to_float(r["wowa_balanced"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(xs))
    ax.bar(x - 0.18, cvar, 0.36, label="CVaR05")
    ax.bar(x + 0.18, wowa, 0.36, label="WOWA balanced")
    ax.set_xticks(x)
    ax.set_xticklabels(xs, rotation=30, ha="right")
    ax.set_title("Diagnostic CVaR/WOWA blindspot")
    ax.set_ylabel("Score")
    ax.legend()
    fig.tight_layout()
    out = Path(out_dir) / "diagnostic_blindspot.svg"
    fig.savefig(out)
    plt.close(fig)
    return out


def save_scaling_curves(results_dir, out_dir):
    files = []
    files += sorted(Path(results_dir).glob("scaling_*.csv"))
    files += sorted(Path(results_dir).glob("large_scale_*.csv"))
    files += sorted(Path(results_dir).glob("large_scale_calibrated_*.csv"))
    files += sorted(Path(results_dir).glob("large_scale_calibrated_multiseed_*_summary.csv"))
    outs = []
    for path in files:
        rows = read_csv(path)
        if not rows:
            continue
        inst = rows[0].get("instance", path.stem)
        # planner-scale summaries use score_mean; raw files use score
        ykey = "score_mean" if "score_mean" in rows[0] else "score"
        planner_key = "planner" if "planner" in rows[0] else None
        if planner_key is None or "budget" not in rows[0]:
            continue
        planners = sorted(set(r[planner_key] for r in rows))
        fig, ax = plt.subplots(figsize=(7, 4))
        for planner in planners:
            pr = [r for r in rows if r[planner_key] == planner]
            pr.sort(key=lambda r: _to_float(r["budget"]))
            x = [_to_float(r["budget"]) for r in pr]
            y = [_to_float(r[ykey]) for r in pr]
            ax.plot(x, y, marker="o", label=planner)
        ax.set_xscale("log")
        ax.set_xlabel("Evaluation budget")
        ax.set_ylabel("Selector score")
        ax.set_title(f"Planner scaling: {inst}")
        ax.legend(fontsize=7)
        fig.tight_layout()
        out = Path(out_dir) / f"scaling_{inst}.svg"
        fig.savefig(out)
        plt.close(fig)
        outs.append(out)
    return outs


def save_baseline_comparison(results_dir, out_dir):
    files = sorted(Path(results_dir).glob("baselines_*.csv"))
    outs = []
    for path in files:
        rows = read_csv(path)
        if not rows:
            continue
        inst = rows[0].get("instance", path.stem)
        xs = [r.get("baseline", "") for r in rows]
        ys = [_to_float(r.get("wowa_balanced", r.get("mean", 0.0))) for r in rows]
        fig, ax = plt.subplots(figsize=(max(8, len(xs) * 0.65), 4))
        ax.bar(range(len(xs)), ys)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(xs, rotation=60, ha="right")
        ax.set_ylabel("WOWA-balanced score")
        ax.set_title(f"Baseline comparison: {inst}")
        fig.tight_layout()
        out = Path(out_dir) / f"baseline_comparison_{inst}.svg"
        fig.savefig(out)
        plt.close(fig)
        outs.append(out)
    return outs


def save_external_sota_status(results_dir, out_dir):
    ext_dir = Path("external_runs")
    files = sorted(ext_dir.glob("sota_*.csv"))
    if not files:
        return []
    outs = []
    for path in files:
        rows = read_csv(path)
        if not rows:
            continue
        inst = rows[0].get("instance", path.stem)
        xs = [r.get("solver", "") for r in rows]
        status_map = {"ran": 1.0, "failed": 0.0, "timeout": -0.5, "skipped_not_installed": -1.0}
        ys = [status_map.get(r.get("status", ""), -1.0) for r in rows]
        fig, ax = plt.subplots(figsize=(max(8, len(xs) * 0.65), 4))
        ax.bar(range(len(xs)), ys)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(xs, rotation=60, ha="right")
        ax.set_ylabel("status")
        ax.set_yticks([-1.0, -0.5, 0.0, 1.0])
        ax.set_yticklabels(["skipped", "timeout", "failed", "ran"])
        ax.set_title(f"External SOTA status: {inst}")
        fig.tight_layout()
        out = Path(out_dir) / f"external_sota_status_{inst}.svg"
        fig.savefig(out)
        plt.close(fig)
        outs.append(out)
    return outs


def save_ordered_loss_profile(results_dir, out_dir, instance):
    summary_path = Path(results_dir) / f"{instance}_summary.csv"
    if not summary_path.exists():
        return None
    rows = read_csv(summary_path)
    if not rows or "selector" not in rows[0]:
        return None
    selectors = ["governed_mean", "governed_cvar05", "governed_wowa_balanced", "oracle_hidden_state"]
    rows = [r for r in rows if r["selector"] in selectors]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    for r in rows:
        profile = [_to_float(r.get(k)) for k in ["worst", "cvar01", "cvar05", "cvar10", "cvar25", "mean"]]
        ax.plot(range(len(profile)), profile, marker="o", label=r["selector"])
    ax.set_xticks(range(6))
    ax.set_xticklabels(["worst", "CVaR01", "CVaR05", "CVaR10", "CVaR25", "mean"])
    ax.set_ylabel("Loss / risk score")
    ax.set_title(f"Ordered-risk proxy profile: {instance}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    out = Path(out_dir) / f"ordered_loss_profiles_{instance}.svg"
    fig.savefig(out)
    plt.close(fig)
    return out


def make_all_figures(results_dir, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    outs = []
    for fn in [save_feasible_fraction, save_oracle_gap, save_governed_selector_comparison, save_diagnostic_blindspot]:
        out = fn(results_dir, out_dir)
        if out:
            outs.append(out)
    outs.extend(save_scaling_curves(results_dir, out_dir))
    outs.extend(save_baseline_comparison(results_dir, out_dir))
    outs.extend(save_external_sota_status(results_dir, out_dir))
    for inst in ["bsp_easy", "bsp_hard", "bcst_easy", "bged_hard", "diagnostic_cvar_blindspot"]:
        out = save_ordered_loss_profile(results_dir, out_dir, inst)
        if out:
            outs.append(out)
    return outs
