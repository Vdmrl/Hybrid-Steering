import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from ready_judge import core


class ReleaseTests(unittest.TestCase):
    def test_theistic_candidate_merges_upper_levels_without_changing_default(self):
        _, rubric, prompts = core.configuration(core.ROOT / "candidates/theistic_v6_1_review1")
        feature = rubric["features"]["theistic_framing"]
        self.assertEqual(rubric["rubric_version"], "6.1.0-theistic-review1")
        self.assertEqual(feature["maximum"], 3)
        self.assertEqual(set(feature["anchors"]), {"0", "1", "2", "3"})
        self.assertNotIn("{{", prompts["theistic_framing"])
        self.assertEqual(core.configuration()[1]["features"]["theistic_framing"]["maximum"], 4)

    def test_concrete_uncertainty_scale_is_not_the_old_four_point_scale(self):
        _, rubric, prompts = core.configuration(core.ROOT / "candidates/concrete_v6_review1")
        self.assertEqual(rubric["rubric_version"], "6.0.0-concrete-review1")
        self.assertEqual(set(rubric["features"]), {"complexity", "probabilistic_framing"})
        for feature in rubric["features"].values():
            self.assertEqual(feature["maximum"], 3)
            self.assertEqual(set(feature["anchors"]), {"0", "1", "2", "3"})
        self.assertNotIn("{{", prompts["probabilistic_framing"])
        with self.assertRaises(ValueError):
            core.parse_choice({"finish_reason": "stop", "message": {"content": "4"}}, 3)
        self.assertEqual(core.configuration()[1]["features"]["probabilistic_framing"]["maximum"], 4)

    def test_explicit_candidate_resources_and_resume_isolation(self):
        for version in ("review1", "review2"):
            _, registry, templates = core.configuration(
                core.ROOT / ("candidates/technical_v5_2_2_" + version)
            )
            self.assertEqual(registry["rubric_version"], "5.2.2-technical-" + version)
            self.assertNotIn("{{", templates["complexity"])
        candidate = core.ROOT / "candidates/technical_v5_2_2_review1"
        _, rubric, prompts = core.configuration(candidate)
        self.assertEqual(rubric["rubric_version"], "5.2.2-technical-review1")
        self.assertEqual(set(rubric["features"]), {"complexity"})
        self.assertEqual(rubric["features"]["complexity"]["maximum"], 3)
        self.assertNotIn("{{concept}}", prompts["complexity"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            source = path / "input.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "prompt_id": "p",
                        "answer_id": "a",
                        "scenario": "Explain.",
                        "text": "Ice melts.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            core.evaluate(source, path / "scores", ["complexity"], resources=candidate)
            manifest = json.loads((path / "scores/manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["identity"]["rubric"], rubric)
            with self.assertRaisesRegex(ValueError, "Resume rejected"):
                core.evaluate(source, path / "scores", ["complexity"])
        self.assertEqual(core.configuration()[1]["rubric_version"], "5.2.1-review1")

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
