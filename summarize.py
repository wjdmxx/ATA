"""Collect run summaries into a CSV; incomplete runs remain explicitly marked."""
import argparse
import csv
import json
from pathlib import Path

import yaml

from utils import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=str(ROOT / "outputs"))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    rows = []
    for file in sorted(root.glob("*/summary.json")):
        config = yaml.safe_load((file.parent / "config.yaml").read_text())
        summary = json.loads(file.read_text())

        rows.append({
            "run": file.parent.name,
            "status": summary["status"],
            "model": config["model"]["name"],
            "method": config["method"]["name"],
            "dataset": config["data"].get("name", "imagenet_c"),
            "group": config["data"]["group"],
            "severities": str(config["data"]["severities"]),
            "samples_per_task": config["data"]["samples"],
            "batch_size": config["data"]["batch_size"],
            "reset": config["protocol"]["reset"],
            "prediction": config["method"]["prediction"],
            "seed": config["run"]["seed"],
            "lr": config["method"]["optimizer"]["lr"],
            "completed_tasks": summary["completed_tasks"],
            "planned_tasks": summary["planned_tasks"],
            "mean_top1": summary.get("mean_top1"),
            "mean_error": summary.get("mean_error"),
            "seconds": summary.get("task_seconds"),
            "peak_memory_mib": summary.get("peak_memory_mib"),
        })

    if not rows:
        print(f"No experiment summaries under {root}")
        return

    output = Path(args.output) if args.output else root / "comparison.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(rows)} runs -> {output}")


if __name__ == "__main__":
    main()
