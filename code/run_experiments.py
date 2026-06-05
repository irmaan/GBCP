from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from sealed_nomdp.instances import make_instance, list_instances
from sealed_nomdp.algorithms import run_selectors

def write_csv(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", choices=list_instances(), default="diagnostic_cvar_blindspot")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--max_sequences", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_all", action="store_true")
    args = ap.parse_args()

    Path("results").mkdir(exist_ok=True)
    names = list_instances() if args.all else [args.instance]
    for name in names:
        env = make_instance(name, args.horizon)
        rows, all_rows = run_selectors(env, max_sequences=args.max_sequences, seed=args.seed, save_all=args.save_all)
        print(f"\n== {name} ==")
        for r in rows:
            print(json.dumps(r, indent=2))
        write_csv(f"results/{name}_summary.csv", rows)
        if args.save_all:
            write_csv(f"results/{name}_all_sequences.csv", all_rows)

if __name__ == "__main__":
    main()
