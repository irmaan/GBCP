# Governed Blind Commitment Planning: SUM 2026 Artifact

This repository contains the artifact for the SUM 2026 paper
**“Governed Blind Commitment Planning under Null-Observation Windows.”**

The artifact supports the paper as an auditable benchmark and certificate package.
It is **not** a planner contribution and does not claim a new POMDP or NOMDP solver.
Its purpose is to reproduce the finite-scenario audits reported in the paper.

## What this artifact reproduces

The artifact contains finite scenario tables, action-sequence losses, harm indicators,
budgets, and scripts for recomputing:

1. deterministic blind feasibility;
2. least-violation certificates;
3. structural versus informational oracle labels;
4. blind-tax values;
5. mixed LP audits;
6. public-template transfer summaries;
7. randomized-generator coverage;
8. perturbation and ablation checks;
9. selected larger-scale search summaries used only as robustness evidence.

## What this artifact does not claim

The search and robustness scripts are included only to reproduce returned-sequence
checks and supporting tables. They are not proposed as a new planner, not compared
as a SOTA planning algorithm, and not used to claim that discounted or external
POMDP solvers directly solve native GBCP.

## Main directories

* `results/`: exact synthetic-core sequence tables and summaries.
* `results_public/`: public-template transfer sequence tables and summaries.
* `exports_lp/`: LP files for mixed blind-commitment audits.
* `paper_tables/`: CSV/TeX tables used to generate paper results.
* `figures/`: paper-supporting figures.
* `code/`: scripts for recomputing the reported audits.
* `sealed_nomdp/`: core implementation of instances, risk selectors, governed audits, and exporters.

## Basic reproduction

Create a Python environment and install standard scientific packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas scipy matplotlib
```

Recompute the exact synthetic-core audits:

```bash
python code/run_experiments.py
```

Recompute mixed LP audits:

```bash
python code/run_exact_mixture_audits.py
```

Recompute public-transfer summaries:

```bash
python code/run_public_sysadmin_transfer.py
python code/run_public_transfer_extra.py
python code/make_public_transfer_safe_tables.py
```

Recompute randomized-generator and ablation summaries:

```bash
python code/random_regime_generator_study_v2.py
python code/run_ablation_table.py
```

Rebuild paper tables:

```bash
python code/make_paper_tables.py
```

## Relation to the paper

The paper’s claims are based on the finite scenario bundles and governed audit
semantics. Once a blind sequence is fixed, feasibility, violation, selector values,
oracle labels, blind tax, and mixed LP audits are recomputed from the released
tables rather than from solver traces. This is why the artifact is organized as a
claim-to-object audit package rather than as a single planner run.

