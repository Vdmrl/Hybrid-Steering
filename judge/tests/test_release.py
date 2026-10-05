import hashlib
import json
import unittest

from ready_judge import core


class ReleaseTests(unittest.TestCase):
    def test_frozen_resources_and_scales(self):
        _, rubric, prompts = core.configuration()
        self.assertEqual(rubric["rubric_version"], "5.2.1-review1")
        scales = {
            "numbered": 4,
            "french": 3,
            "complexity": 3,
            "theistic_framing": 4,
            "probabilistic_framing": 4,
            "fairy_tale": 4,
        }
        self.assertEqual({f: r["maximum"] for f, r in rubric["features"].items()}, scales)
        provenance = json.loads(
            (core.ROOT.parents[1] / "calibration/release-5.2.1-provenance.json").read_text(
                encoding="utf-8"
            )
        )
        for name, digest in provenance["resources_sha256"].items():
            self.assertEqual(
                hashlib.sha256((core.ROOT / name).read_bytes()).hexdigest(), digest, name
            )
        self.assertIn("Never translate RESPONSE mentally", prompts["french"])
        self.assertIn("small LOCAL".lower(), prompts["complexity"].lower())

    def test_calibration_fixtures_fit_actual_scales(self):
        _, rubric, _ = core.configuration()
        calibration = core.ROOT.parents[1] / "calibration"
        for directory in ("five-traits-v52", "french-v521"):
            cases = json.loads(
                (calibration / directory / "cases-private.json").read_text(encoding="utf-8")
            )
            self.assertTrue(cases)
            for case in cases:
                core.normalized(case["gold_score"], rubric["features"][case["feature"]]["maximum"])
        historical = json.loads(
            (core.ROOT / "concepts/features_v5_0_0_rc1.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(historical["rubric_version"], "5.0.0-rc1")
        self.assertEqual(historical["features"]["complexity"]["maximum"], 2)
