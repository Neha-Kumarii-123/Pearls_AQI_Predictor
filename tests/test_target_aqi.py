import sys
import unittest
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MODULE_PATH = ROOT / "src" / "backfill_pipeline.py"
spec = importlib.util.spec_from_file_location("backfill_pipeline", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TargetAQITest(unittest.TestCase):
    def test_target_aqi_stays_direct_and_continuous(self):
        sample = module.fetch_historical_data(
            start_date="2024-08-01",
            end_date="2024-08-02"
        )
        self.assertIn("target_aqi", sample.columns)
        self.assertTrue(sample["target_aqi"].notna().all())
        self.assertTrue(sample["target_aqi"].dtype.kind in {"f", "i"})


if __name__ == "__main__":
    unittest.main()
