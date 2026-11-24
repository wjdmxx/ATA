"""Two-machine path selection without requiring either machine's datasets."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import load_config, resolve_machine_path


class MachinePathTests(unittest.TestCase):
    def test_primary_then_fallback_then_error(self):
        with tempfile.TemporaryDirectory() as scratch:
            primary = Path(scratch) / "4090"
            fallback = Path(scratch) / "h20"
            with patch.dict("utils.MACHINE_PATHS", {primary: fallback}, clear=True):
                with self.assertRaises(FileNotFoundError) as error:
                    resolve_machine_path(primary, required=True)
                self.assertIn(str(primary), str(error.exception))
                self.assertIn(str(fallback), str(error.exception))
                # Dry runs can still describe an unavailable dataset.
                self.assertEqual(resolve_machine_path(primary), primary)
                fallback.mkdir()
                self.assertEqual(resolve_machine_path(primary, required=True), fallback)
                primary.mkdir()
                self.assertEqual(resolve_machine_path(primary, required=True), primary)

    def test_holder_script_and_state_directory_follow_same_priority(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            primary_script, fallback_script = root / "g.py", root / "gpu_hold.py"
            primary_state, fallback_state = root / "run", root / "tmp"
            fallback_script.touch()
            fallback_state.mkdir()
            with patch.dict("utils.MACHINE_PATHS", {
                primary_script: fallback_script, primary_state: fallback_state,
            }, clear=True):
                self.assertEqual(resolve_machine_path(primary_script, required=True), fallback_script)
                self.assertEqual(resolve_machine_path(primary_state, required=True), fallback_state)
                custom = root / "custom.py"
                with self.assertRaises(FileNotFoundError):
                    resolve_machine_path(custom, required=True)
                custom.touch()
                self.assertEqual(resolve_machine_path(custom, required=True), custom)

    def test_config_resolves_corruption_clean_and_fisher_paths(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            corruption, clean = root / "imagenet-c", root / "val"
            corruption.mkdir()
            clean.mkdir()
            defaults = {
                Path("/mnt/bit/data/imagenet-c"): corruption,
                Path("/mnt/bit/data/imagenet/images/val"): clean,
            }
            # Make the primary paths absent independently of the host machine.
            exists = Path.exists
            def available(path):
                return False if path in defaults else exists(path)
            with patch.dict("utils.MACHINE_PATHS", defaults, clear=True), \
                    patch.object(Path, "exists", available):
                config = load_config()
                self.assertEqual(config["data"]["root"], str(corruption))
                self.assertEqual(config["method"]["eata"]["fisher_root"], str(clean))
                config = load_config([ROOT / "configs/imagenet_50k.yaml"])
                self.assertEqual(config["data"]["root"], str(clean))
                config = load_config(overrides=[f"data.root={root}"])
                self.assertEqual(config["data"]["root"], str(root))


if __name__ == "__main__":
    unittest.main()
