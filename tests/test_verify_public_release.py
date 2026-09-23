from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import verify_public_release as release


ROOT = Path(__file__).resolve().parents[1]


class PublicReleaseVerificationTest(unittest.TestCase):
    def test_current_public_release_passes_without_private_inputs(self) -> None:
        summary = release.verify_release(ROOT)
        self.assertEqual(summary["charts"], 1_023)
        self.assertEqual(summary["global_scores"], 28_644)
        self.assertEqual(summary["sequence_scores"], 8_184)
        self.assertEqual(summary["sequence_fold_runs"], 80)
        self.assertEqual(summary["dojo_placements_per_family"], 47)

    def test_manifest_rejects_changed_bytes_and_extra_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            payload = directory / "sample.json"
            payload.write_text("{}\n", encoding="utf-8")
            privacy = directory / "privacy_report.json"
            privacy.write_text(
                json.dumps({"status": "pass", "forbidden_findings": 0}),
                encoding="utf-8",
            )
            files = {
                path.name: {"sha256": release.sha256(path), "size_bytes": path.stat().st_size}
                for path in (payload, privacy)
            }
            (directory / "MANIFEST.json").write_text(
                json.dumps({"schema_version": release.MANIFEST_SCHEMA, "files": files}),
                encoding="utf-8",
            )
            with patch.object(release, "REQUIRED_FILES", frozenset(files)):
                self.assertEqual(release.verify_manifest(directory), 3)
                payload.write_text('{"changed": true}\n', encoding="utf-8")
                with self.assertRaisesRegex(release.ReleaseError, "mismatch"):
                    release.verify_manifest(directory)
                payload.write_text("{}\n", encoding="utf-8")
                (directory / "extra.txt").write_text("unexpected", encoding="utf-8")
                with self.assertRaisesRegex(release.ReleaseError, "missing or extra"):
                    release.verify_manifest(directory)

    def test_privacy_scan_rejects_structured_identity_and_private_path(self) -> None:
        release.check_private_content({"chart_titles": "excluded"})
        with self.assertRaisesRegex(release.ReleaseError, "private field"):
            release.check_private_content({"nested": [{"normalized_title": "example"}]})
        with self.assertRaisesRegex(release.ReleaseError, "private path"):
            release.check_private_content({"source": "/Users/example/private.json"})

    def test_dojo_score_must_match_native_out_of_fold_prediction(self) -> None:
        directory = ROOT / release.ARTIFACT
        reference, global_index, sequence_index, _ = release.verify_scores(directory)
        row = release.read_jsonl(directory / "dojo_public_matches.jsonl")[0]
        score = global_index["native", row["condition"], row["chart_key"]]
        global_index["native", row["condition"], row["chart_key"]] = {
            **score,
            "latent_score": score["latent_score"] + 1,
        }
        with self.assertRaisesRegex(release.ReleaseError, "Dojo score differs"):
            release.verify_dojo(directory, reference, global_index, sequence_index)


if __name__ == "__main__":
    unittest.main()
