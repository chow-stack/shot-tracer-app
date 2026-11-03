from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "verify_ios_parity_fixtures.py"
SPEC = importlib.util.spec_from_file_location("verify_ios_parity_fixtures", SCRIPT_PATH)
fixtures = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(fixtures)


class IOSParityFixtureTests(unittest.TestCase):
    def test_frozen_python_reference_artifacts_match_manifest(self):
        results = fixtures.verify_manifest()
        self.assertEqual(len(results), 5)
        self.assertTrue(all(result["status"] == "verified" for result in results))


if __name__ == "__main__":
    unittest.main()
