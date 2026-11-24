"""Run one ViTTT classification / TTA experiment."""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import ImageNetImages, build_fisher_loader, build_transform, get_tasks, make_manifest, task_folder
from methods import build_method
from models import build_model, download_weights, load_weights
from utils import (
    ROOT, atomic_save, capture_rng, environment_info, load_config,
    make_logger, make_run_dir, restore_rng, resume_config,
    seed_everything, sha256_file, snapshot_code, write_json,
)

ADAPTATION_COUNTS = ("n_reliable", "n_selected", "forward_calls", "backward_calls", "recoveries")


def should_reset(mode, index, tasks):
    if mode == "task":
        return True
    if mode == "domain":
        return index == 0 or tasks[index][0] != tasks[index - 1][0]
    return False


def write_results(directory, rows):
    if not rows:
        return

    with (directory / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    if not rows:
        return {"completed_tasks": 0}

    count = sum(row["samples"] for row in rows)
    mean_top1 = float(np.mean([row["top1"] for row in rows]))
    by_domain = {}
    for row in rows:
        by_domain.setdefault(row["domain"], []).append(row["top1"])

    return {
        "completed_tasks": len(rows),
        "evaluated_images": count,
        "mean_top1": mean_top1,
        "mean_error": 100.0 - mean_top1,
        "weighted_top1": 100.0 * sum(row["correct1"] for row in rows) / count,
        "mean_top5": float(np.mean([row["top5"] for row in rows])),
        "task_seconds": sum(row["seconds"] for row in rows),
        "peak_memory_mib": max(row["peak_memory_mib"] for row in rows),
        "by_domain": {name: float(np.mean(values)) for name, values in by_domain.items()},
        **{key: sum(row.get(key, 0) for row in rows) for key in ADAPTATION_COUNTS if key in rows[0]},
    }


def report_top1(directory, summary):
    names = list(summary["by_domain"])
    scores = list(summary["by_domain"].values())
    if len(scores) > 1:
        names.append("mean")
        scores.append(summary["mean_top1"])

    order = "Top-1 order: " + " ".join(names)
    line = " ".join(f"{score:.2f}" for score in scores)
    (directory / "top1.txt").write_text(line + "\n")

    # Keep the copyable line free of the usual logger timestamp and labels.
    with (directory / "run.log").open("a") as handle:
        handle.write(order + "\n" + line + "\n")
    print(order, flush=True)
    print(line, flush=True)
    return line


def restore_checkpoint(path, method, config, manifest_hash, source_hash, device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if resume_config(config) != resume_config(state["config"]):
        raise ValueError("Resume configuration differs from the saved experiment")
    if state["manifest_sha256"] != manifest_hash:
        raise ValueError("Resume sample identities/order differ")
    if state["source_sha256"] != source_hash:
        raise ValueError("Source checkpoint changed since the saved experiment")

    method.load_state_dict(state["method"])
    restore_rng(state["rng"], device)
    return state["results"], state["next_task"]


def save_checkpoint(directory, method, config, rows, manifest_hash, source_hash, device):
    mode = config["run"]["save_weights"]
    if mode == "none":
        return

    method_state = method.state_dict()
    state = {
        "format": "ata-task-checkpoint-v1",
        "config": config,
        "model": method_state["model"],
        "method": method_state,
        "results": rows,
        "next_task": len(rows),
        "source_sha256": source_hash,
        "manifest_sha256": manifest_hash,
        "rng": capture_rng(device),
    }
    atomic_save(directory / "checkpoints/last.pt", state)

    if mode == "task":
        domain, severity = rows[-1]["domain"], rows[-1]["severity"]
        name = domain if severity is None else f"{domain}_s{severity}"
        atomic_save(directory / f"checkpoints/{name}.pt", state)


def evaluate_task(method, loader, device, config, directory, task, task_index, logger):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    count = correct1 = correct5 = updates = 0
    model_seconds = 0.0
    adaptation_counts = {}

    with (directory / "metrics.jsonl").open("a") as metrics_file:
        for batch_index, (images, labels) in enumerate(loader):
            if config["protocol"]["reset"] == "batch":
                method.reset()

            batch_started = time.perf_counter()
            images = images.to(device, non_blocking=device.type == "cuda")

            # Labels stay in the evaluator; adaptation only receives images.
            logits, adaptation_metrics = method.step(images)
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Non-finite predictions")
            if logits.shape != (len(labels), 1000):
                raise ValueError(f"Expected logits [B,1000], got {tuple(logits.shape)}")

            predicted = logits.topk(5, dim=1).indices.cpu()
            elapsed = time.perf_counter() - batch_started
            batch_correct1 = int((predicted[:, 0] == labels).sum())
            batch_correct5 = int((predicted == labels[:, None]).any(dim=1).sum())

            count += len(labels)
            correct1 += batch_correct1
            correct5 += batch_correct5
            updates += adaptation_metrics.get("updates", 0)
            for key in ADAPTATION_COUNTS:
                if key in adaptation_metrics:
                    adaptation_counts[key] = adaptation_counts.get(key, 0) + adaptation_metrics[key]
            model_seconds += elapsed

            record = {
                "task": task_index,
                "domain": task[0],
                "severity": task[1],
                "batch": batch_index,
                "images_seen": count,
                "batch_top1": 100.0 * batch_correct1 / len(labels),
                "running_top1": 100.0 * correct1 / count,
                "model_seconds": elapsed,
                **adaptation_metrics,
            }
            metrics_file.write(json.dumps(record, allow_nan=False) + "\n")

            if (batch_index + 1) % config["run"]["log_every"] == 0:
                metrics_file.flush()
                logger.info(
                    "%s/%s batch %d/%d images %d top1 %.3f",
                    *task, batch_index + 1, len(loader), count, record["running_top1"],
                )

    peak_memory = 0.0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory = torch.cuda.max_memory_allocated(device) / 2**20

    return {
        "domain": task[0],
        "severity": task[1],
        "samples": count,
        "correct1": correct1,
        "correct5": correct5,
        "top1": 100.0 * correct1 / count,
        "top5": 100.0 * correct5 / count,
        "error": 100.0 * (1.0 - correct1 / count),
        "updates": updates,
        "seconds": time.perf_counter() - started,
        "model_seconds": model_seconds,
        "peak_memory_mib": peak_memory,
        **adaptation_counts,
    }


def run(config, resume=None):
    data_config = config["data"]
    run_config = config["run"]
    device = torch.device(run_config["device"])
    if device.type == "cuda":
        torch.cuda.set_device(device)

    # Fix the data stream before creating the model or adapting it.
    tasks = get_tasks(data_config)
    for task in tasks:
        folder = task_folder(data_config, *task)
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing task: {folder}")

    seed = run_config["seed"]
    seed_everything(seed, run_config["deterministic"])
    manifest = make_manifest(data_config, seed=seed)
    manifest_hash = manifest["sha256"]

    # Load the source model; the method keeps the initial state for resets.
    checkpoint_path = Path(config["model"]["checkpoint"])
    source_hash = sha256_file(checkpoint_path)
    model = build_model(config["model"]["name"])
    load_weights(model, checkpoint_path, config["model"]["checkpoint_key"])
    method = build_method(model.to(device), config["method"])

    rows, first_task = [], 0
    if resume:
        rows, first_task = restore_checkpoint(
            resume, method, config, manifest_hash, source_hash, device,
        )

    # Each experiment owns its config, code snapshot, samples, logs and results.
    directory = make_run_dir(config)
    logger = make_logger(directory)
    metadata = {
        "status": "running",
        "environment": environment_info(device),
        "source_checkpoint": str(checkpoint_path),
        "source_sha256": source_hash,
        "sample_sha256": manifest_hash,
        "tasks": tasks,
        "dataset": data_config["name"],
        "seed": seed,
        "method": config["method"]["name"],
        "precision": "float32",
        "parameter_names": method.parameter_names,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "resumed_from": str(Path(resume).resolve()) if resume else None,
        "source_code_sha256": snapshot_code(directory),
    }
    summary = {
        "status": "running",
        "dataset": data_config["name"],
        "planned_tasks": len(tasks),
        **summarize(rows),
    }
    write_json(directory / "metadata.json", metadata)
    write_json(directory / "samples.json", manifest)
    write_results(directory, rows)
    logger.info(
        "Run: %s | %d tasks | %d images/task | method=%s | device=%s",
        directory, len(tasks), len(manifest["samples"]), config["method"]["name"], device,
    )

    started = time.perf_counter()
    try:
        if config["method"]["name"] == "eata":
            fisher_started = time.perf_counter()
            fisher_loader, fisher_manifest = build_fisher_loader(data_config, config["method"]["eata"], seed)
            write_json(directory / "fisher_samples.json", fisher_manifest)
            method.estimate_fisher(fisher_loader)
            metadata["fisher"] = {"samples": len(fisher_loader.dataset), "batches": len(fisher_loader),
                                  "sample_sha256": fisher_manifest["sha256"],
                                  "seconds": time.perf_counter() - fisher_started}
            write_json(directory / "metadata.json", metadata)
            logger.info("Fisher: %d clean images, %d batches, %.2f seconds",
                        metadata["fisher"]["samples"], metadata["fisher"]["batches"], metadata["fisher"]["seconds"])

        transform = build_transform(data_config)
        for index in range(first_task, len(tasks)):
            task = tasks[index]
            if should_reset(config["protocol"]["reset"], index, tasks):
                method.reset()

            folder = task_folder(data_config, *task)
            dataset = ImageNetImages(folder, manifest, transform)
            generator = torch.Generator().manual_seed(seed + index)
            loader = DataLoader(
                dataset,
                batch_size=data_config["batch_size"],
                num_workers=data_config["workers"],
                shuffle=False,
                drop_last=False,
                pin_memory=device.type == "cuda",
                generator=generator,
            )

            logger.info("Task %d/%d: %s severity=%s", index + 1, len(tasks), *task)
            row = evaluate_task(method, loader, device, config, directory, task, index, logger)
            rows.append(row)

            write_results(directory, rows)
            summary.update(summarize(rows))
            write_json(directory / "summary.json", summary)
            logger.info(
                "Completed %s/%s: top1=%.3f top5=%.3f images=%d",
                *task, row["top1"], row["top5"], row["samples"],
            )

            save_checkpoint(directory, method, config, rows, manifest_hash, source_hash, device)

        summary.update(status="completed", this_process_seconds=time.perf_counter() - started)
        metadata["status"] = "completed"
        write_json(directory / "summary.json", summary)
        write_json(directory / "metadata.json", metadata)
        logger.info(
            "Completed: mean top1 %.3f, mean error %.3f",
            summary["mean_top1"], summary["mean_error"],
        )
        report_top1(directory, summary)

    except BaseException as error:
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        metadata.update(status=status, error=f"{type(error).__name__}: {error}")
        summary.update(status=status, **summarize(rows))
        write_json(directory / "metadata.json", metadata)
        write_json(directory / "summary.json", summary)
        logger.exception("Experiment stopped")
        raise

    finally:
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

    return directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", nargs="+", default=[], help="YAML overlays on base.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="Dotted YAML overrides")
    parser.add_argument("--dry-run", action="store_true", help="Print configuration and task plan only")
    parser.add_argument("--resume", help="Resume from an ATA task checkpoint into a new run directory")
    parser.add_argument("--download", nargs="+", choices=("tiny", "small", "base"), help="Download weights and exit")
    args = parser.parse_args(argv)

    if args.download:
        for name in args.download:
            print(download_weights(name, ROOT / "weights/pretrained"))
        return

    config = load_config(args.config, args.set)
    if args.dry_run:
        tasks = get_tasks(config["data"])
        plan = {
            "config": config,
            "tasks": tasks,
            "num_tasks": len(tasks),
            "images_per_task": config["data"]["samples"],
            "total_image_evaluations": len(tasks) * config["data"]["samples"],
            "checkpoint_exists": Path(config["model"]["checkpoint"]).is_file(),
            "data_root_exists": Path(config["data"]["root"]).is_dir(),
        }
        print(json.dumps(plan, indent=2))
        return

    return run(config, args.resume)


if __name__ == "__main__":
    main()
