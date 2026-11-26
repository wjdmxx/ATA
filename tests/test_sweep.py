"""Check GPU selection, the baseline grid, and holder/direct execution."""
from contextlib import redirect_stderr, redirect_stdout
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
    def test_baseline_grid_has_36_validation_runs_on_selected_gpus(self):
        settings = yaml.safe_load((ROOT / "configs/sweep_baselines_val_50k.yaml").read_text())
        settings["gpus"] = [1, 2, 4]
        jobs = sweep.make_plan(settings, Path("/tmp/ata-test-plan"))
        rates = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
        expected = set(itertools.product(("small", "base"), ("eata", "sar", "deyo"), rates, (0,)))
        self.assertEqual(len(jobs), 36)
        self.assertEqual({(j["model"], j["method"], j["lr"], j["seed"]) for j in jobs}, expected)
        self.assertEqual({job["gpu"] for job in jobs}, {1, 2, 4})
        for job in jobs:
            config = load_config(job["configs"], job["overrides"])
            self.assertEqual(get_tasks(config["data"]), [(domain, 5) for domain in EXTRA])
            self.assertEqual(config["data"]["samples"], 50000)
            self.assertEqual(config["data"]["batch_size"], 64)
            self.assertEqual(config["method"]["eata"]["fisher_size"], 2000)
            self.assertEqual(config["run"]["device"], f"cuda:{job['gpu']}")
            self.assertEqual(config["protocol"]["reset"], "task")

    def test_cli_selects_subset_or_all_visible_gpus(self):
        config = ROOT / "configs/sweep_baselines_val_50k.yaml"
        for requested, count, expected in ((["2", "5"], 8, [2, 5]),
                                           (["all"], 2, [0, 1]),
                                           (["all"], 8, list(range(8)))):
            with self.subTest(requested=requested, visible=count):
                argv = ["sweep.py", "--config", str(config), "--gpus", *requested, "--dry-run"]
                with patch.object(sys, "argv", argv), \
                        patch("sweep.torch.cuda.device_count", return_value=count) as device_count, \
                        patch("sweep.make_plan", wraps=sweep.make_plan) as make_plan, \
                        redirect_stdout(io.StringIO()):
                    self.assertEqual(sweep.main(), 0)
                self.assertEqual(make_plan.call_args.args[0]["gpus"], expected)
                self.assertEqual(device_count.call_count, int(requested == ["all"]))

    def test_gpu_selection_is_required_without_a_legacy_config_list(self):
        argv = ["sweep.py", "--config", str(ROOT / "configs/sweep_baselines_val_50k.yaml"), "--dry-run"]
        with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()), \
                redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            sweep.main()
        self.assertEqual(error.exception.code, 2)

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

    def test_holder_mode_wraps_command_and_respects_explicit_state_directory(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            settings = root / "sweep.yaml"
            settings.write_text("seeds: [0]\n")
            checkpoint = root / "source.pth"
            checkpoint.touch()
            holder = root / "resolved_holder.py"
            command = [sys.executable, "-S", "-c", "pass"]
            job = dict(name="job", seed=0, gpu=2, estimated_minutes=0,
                       configs=[], overrides=[], status="pending", command=command)
            config = {"model": {"checkpoint": str(checkpoint)},
                      "data": {"name": "imagenet", "group": "clean", "domains": None,
                               "severities": [], "root": str(root)}}
            output = root / "queue"
            configured_holder = str(root / "configured_holder.py")
            state_directory = str(root / "holder-state")
            argv = ["sweep.py", "--config", str(settings), "--output", str(output),
                    "--gpus", "2", "--holder", configured_holder, "--controller-pid", "123"]
            with patch.object(sys, "argv", argv), \
                    patch.dict("os.environ", GPU_HOLDER_STATE_DIR=state_directory), \
                    patch("sweep.make_plan", return_value=[job]), \
                    patch("sweep.load_config", return_value=config), \
                    patch("sweep.resolve_machine_path", return_value=holder) as resolve_path, \
                    patch("sweep.subprocess.check_output", return_value="test-revision\n"), \
                    patch("sweep.subprocess.Popen") as popen, \
                    patch("sweep.signal.signal"), patch("sweep.subprocess.run"), \
                    redirect_stdout(io.StringIO()):
                popen.return_value.wait.return_value = 0
                self.assertEqual(sweep.main(), 0)
            resolve_path.assert_called_once_with(configured_holder, required=True)
            self.assertEqual(popen.call_args.args[0], [sys.executable, "-B", "-u", str(holder),
                             "exec", "--gpus", "2", "--controller-pid", "123", "--", *command])
            self.assertEqual(popen.call_args.kwargs["env"]["GPU_HOLDER_STATE_DIR"], state_directory)
            status = json.loads((output / "status.json").read_text())
            self.assertTrue(status["use_holder"])
            self.assertEqual(status["jobs"][0]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
