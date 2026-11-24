"""Balance independent experiments across GPUs, queuing seeds in order."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from decimal import Decimal
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import threading
from zoneinfo import ZoneInfo

import yaml

from data import get_tasks, task_folder
from utils import ROOT, load_config, resolve_machine_path


def make_plan(settings, output):
    jobs = []
    for seed in settings["seeds"]:
        stage = []
        for experiment in settings["experiments"]:
            model, method = experiment["model"], experiment["method"]
            size = {"base": "b", "small": "s", "tiny": "t"}[model]
            for lr in experiment["learning_rates"]:
                for split, data_config in settings["data_configs"].items():
                    name = f"vittt_{size}_{method}_sgd{lr:g}_{split}50k_seed{seed}"
                    configs = [str(ROOT / data_config), str(ROOT / f"configs/{method}.yaml")]
                    overrides = [f"model.name={model}",
                                 f"model.checkpoint=weights/pretrained/vittt_{model}.pth",
                                 "method.optimizer.name=sgd", f"method.optimizer.lr={Decimal(str(lr)):f}",
                                 "method.optimizer.momentum=0.9", "method.optimizer.weight_decay=0.0",
                                 "data.batch_size=64", "run.save_weights=none",
                                 f"run.seed={seed}", f"run.name={name}", f"run.output={output / 'runs'}"]
                    config = load_config(configs, overrides)
                    stage.append({"name": name, "model": model, "method": method,
                                  "lr": lr, "seed": seed, "split": split,
                                  "estimated_minutes": len(get_tasks(config["data"])) * experiment["minutes_per_domain"],
                                  "configs": configs, "overrides": overrides, "status": "pending"})

        loads = {gpu: 0.0 for gpu in settings["gpus"]}
        for job in sorted(stage, key=lambda item: -item["estimated_minutes"]):
            gpu = min(loads, key=loads.get)
            job["gpu"] = gpu
            job["overrides"].append(f"run.device=cuda:{gpu}")
            job["command"] = [sys.executable, "-B", "-u", str(ROOT / "run.py"),
                              "--config", *job["configs"], "--set", *job["overrides"]]
            loads[gpu] += job["estimated_minutes"]
            jobs.append(job)
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/sweep_50k.yaml"))
    parser.add_argument("--output")
    parser.add_argument("--holder", default=str(ROOT.parent / "g/g.py"))
    parser.add_argument("--controller-pid", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    settings = yaml.safe_load(Path(args.config).read_text())
    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")
    output = Path(args.output or ROOT / "outputs" / f"sweep_{timestamp}").resolve()
    jobs = make_plan(settings, output)
    print(f"{len(jobs)} experiments; seeds {settings['seeds']}; GPUs {settings['gpus']}", flush=True)
    for seed in settings["seeds"]:
        for gpu in settings["gpus"]:
            queue = [job for job in jobs if job["seed"] == seed and job["gpu"] == gpu]
            print(f"seed={seed} GPU={gpu} estimated={sum(j['estimated_minutes'] for j in queue):.1f} min: "
                  + ", ".join(job["name"] for job in queue), flush=True)
    if args.dry_run:
        return 0

    args.holder = str(resolve_machine_path(args.holder, required=True))

    # Check only requested task folders and weights before releasing GPUs.
    for job in jobs:
        config = load_config(job["configs"], job["overrides"])
        for path in [Path(config["model"]["checkpoint"]),
                     *(task_folder(config["data"], *task) for task in get_tasks(config["data"]))]:
            if not path.exists():
                raise FileNotFoundError(path)

    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    (output / "sweep.yaml").write_text(yaml.safe_dump(settings, sort_keys=False))
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    lock = threading.RLock()
    stop = threading.Event()
    active = set()

    def save_status():
        temporary = output / "status.tmp"
        temporary.write_text(json.dumps({"git_commit": revision, "jobs": jobs}, indent=2) + "\n")
        temporary.replace(output / "status.json")

    def message(text):
        print(f"{datetime.now().isoformat(timespec='seconds')} {text}", flush=True)

    def request_stop(signum, frame):
        stop.set()
        with lock:
            for process in active:
                process.send_signal(signal.SIGTERM)

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, request_stop)

    environment = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    if "GPU_HOLDER_STATE_DIR" not in environment:
        environment["GPU_HOLDER_STATE_DIR"] = str(resolve_machine_path(
            f"/run/user/{os.getuid()}/gpu-holder", required=True))
    save_status()

    def run_gpu(gpu):
        for job in [j for j in jobs if j["gpu"] == gpu]:
            with lock:
                if stop.is_set():
                    return
                job.update(status="running", started_at=datetime.now().isoformat())
                save_status()
                message(f"START GPU {gpu}: {job['name']}")

            command = [sys.executable, "-B", "-u", args.holder, "exec", "--gpus", str(gpu)]
            if args.controller_pid is not None:
                command += ["--controller-pid", str(args.controller_pid)]
            command += ["--", *job["command"]]
            with (output / "logs" / f"{job['name']}.log").open("w") as log:
                log.write(shlex.join(command) + "\n")
                log.flush()
                with lock:
                    if stop.is_set():
                        job["status"] = "interrupted"
                        save_status()
                        return
                    process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                               stdout=log, stderr=subprocess.STDOUT)
                    active.add(process)
                returncode = process.wait()

            with lock:
                active.remove(process)
                status = "interrupted" if stop.is_set() else "completed" if returncode == 0 else "failed"
                job.update(status=status, returncode=returncode, finished_at=datetime.now().isoformat())
                matches = sorted((output / "runs").glob(f"*_{job['name']}_*"))
                job["run_directory"] = str(matches[-1]) if matches else None
                save_status()
                message(f"{status.upper()} GPU {gpu}: {job['name']} (exit {returncode})")

        subprocess.run([sys.executable, "-B", str(ROOT / "summarize.py"), str(output / "runs")], check=True)

    with ThreadPoolExecutor(max_workers=len(settings["gpus"])) as pool:
        futures = [pool.submit(run_gpu, gpu) for gpu in settings["gpus"]]
        for future in futures:
            future.result()

    counts = {status: sum(job["status"] == status for job in jobs)
              for status in ("pending", "running", "completed", "failed", "interrupted")}
    message(f"QUEUE FINISHED: {counts}")
    return int(any(job["status"] != "completed" for job in jobs))


if __name__ == "__main__":
    raise SystemExit(main())
