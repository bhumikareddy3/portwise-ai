"""CLI: (re)train, evaluate on the chronological hold-out, and write metrics to disk.

    python train.py                      # default 12 h threshold
    python train.py --threshold 24       # different congestion definition
    python train.py --regenerate         # regenerate the synthetic CSV first
"""
from __future__ import annotations

import argparse
import json

import config
from data_generator import load_or_generate
from model import load_or_train, metrics_table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=config.CONGESTION_THRESHOLD_HOURS)
    ap.add_argument("--regenerate", action="store_true")
    a = ap.parse_args()

    df = load_or_generate(force=a.regenerate)
    b = load_or_train(df, a.threshold, force=True)
    tbl = metrics_table(b)

    print(f"Records: {len(df)} | train: {b.n_train} | test: {b.n_test} (from {b.cutoff:%Y-%m-%d})")
    print(f"Congestion = actual delay > {a.threshold:g} h | base rate train {b.metrics['base_rate_train']:.1%}, "
          f"test {b.metrics['base_rate_test']:.1%}\n")
    print(tbl.to_string(index=False))
    for k in ("lr", "rf", "baseline"):
        print(f"\nConfusion matrix [{k}] (rows=actual 0/1, cols=predicted 0/1): {b.metrics[k]['confusion_matrix']}")

    config.REPORT_DIR.mkdir(exist_ok=True)
    (config.REPORT_DIR / "metrics.json").write_text(json.dumps(b.metrics, indent=2))
    (config.REPORT_DIR / "results.md").write_text(
        f"# Results (seed {config.SEED}, threshold {a.threshold:g} h)\n\n"
        f"Chronological split: {b.n_train} train / {b.n_test} test (test starts {b.cutoff:%Y-%m-%d}).\n\n"
        + tbl.to_markdown(index=False) + "\n")
    print(f"\nSaved reports/metrics.json and reports/results.md")


if __name__ == "__main__":
    main()
