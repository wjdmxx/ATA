"""Check the baseline grid and execution without a running GPU holder."""
from contextlib import redirect_stdout
import io
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data import EXTRA, get_tasks
import sweep
from utils import load_config


class SweepTests(unittest.TestCase):
    def test_baseline_grid_has_36_validation_runs_on_eight_gpus(self):
        settings = yaml.safe_load((ROOT / "configs/sweep_baselines_val_50k.yaml").read_text())
        jobs = sweep.make_plan(settings, Path("/tmp/ata-test-plan"))
        rates = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
        expected = set(itertools.product(("small", "base"), ("eata", "sar", "deyo"), rates, (0,)))
        self.assertEqual(len(jobs), 36)
        self.assertEqual({(j["model"], j["method"], j["lr"], j["seed"]) for j in jobs}, expected)
        self.assertEqual({job["gpu"] for job in jobs}, set(range(8)))
        for job in jobs:
            config = load_config(job["configs"], job["overrides"])
            self.assertEqual(get_tasks(config["data"]), [(domain, 5) for domain in EXTRA])
            self.assertEqual(config["data"]["samples"], 50000)
            self.assertEqual(config["data"]["batch_size"], 64)
            self.assertEqual(config["method"]["eata"]["fisher_size"], 2000)
            self.assertEqual(config["run"]["device"], f"cuda:{job['gpu']}")
            self.assertEqual(config["protocol"]["reset"], "task")

    def test_direct_queue_runs_without_holder_and_continues_after_failure(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            settings = root / "sweep.yaml"
            settings.write_text("gpus: [0]\nseeds: [0]\n")
            checkpoint = root / "source.pth"
            checkpoint.touch()
            marker = root / "ran.txt"
            commands = [
                [sys.executable, "-S", "-c", "raise SystemExit(3)"],
                [sys.executable, "-S", "-c", f"from pathlib import Path; Path({str(marker)!r}).write_text('ok')"],
            ]
            jobs = [dict(name=f"job{i}", seed=0, gpu=0, estimated_minutes=0,
                         configs=[], overrides=[], status="pending", command=command)
                    for i, command in enumerate(commands)]
            config = {"model": {"checkpoint": str(checkpoint)},
                      "data": {"name": "imagenet", "group": "clean", "domains": None,
                               "severities": [], "root": str(root)}}
            output = root / "queue"
            argv = ["sweep.py", "--config", str(settings), "--output", str(output),
                    "--no-holder", "--holder", str(root / "missing_holder.py")]
            with patch.object(sys, "argv", argv), \
                    patch("sweep.make_plan", return_value=jobs), \
                    patch("sweep.load_config", return_value=config), \
                    patch("sweep.resolve_machine_path", side_effect=AssertionError("Holder lookup")), \
                    patch("sweep.subprocess.check_output", return_value="test-revision\n"), \
                    patch("sweep.signal.signal"), patch("sweep.subprocess.run"), \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(sweep.main(), 1)
            self.assertEqual(marker.read_text(), "ok")
            status = json.loads((output / "status.json").read_text())
            self.assertFalse(status["use_holder"])
            self.assertEqual([job["status"] for job in status["jobs"]], ["failed", "completed"])
            self.assertEqual([job["returncode"] for job in status["jobs"]], [3, 0])


if __name__ == "__main__":
    unittest.main()
