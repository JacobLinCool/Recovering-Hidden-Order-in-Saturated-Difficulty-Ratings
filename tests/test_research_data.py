from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from TaikoChartEstimator.research.evidence import research_source_snapshot
from TaikoChartEstimator.research.data import (
    MANIFEST_SCHEMA_VERSION,
    ManifestError,
    assign_primary_split,
    normalize_title,
    stable_fraction,
    validate_dataset_manifest,
)


class ResearchDataTest(unittest.TestCase):
    def test_normalize_title_is_unicode_safe(self) -> None:
        self.assertEqual(normalize_title(" ＡＢＣ・太鼓! "), "abc太鼓")
        self.assertEqual(normalize_title("Straße"), "strasse")

    def test_split_assignment_is_stable(self) -> None:
        key = "circleofseasons"
        self.assertEqual(stable_fraction(key, 2027), stable_fraction(key, 2027))
        self.assertEqual(
            assign_primary_split(key, 2027),
            assign_primary_split(key, 2027),
        )

    def test_manifest_rejects_cross_split_title_leakage(self) -> None:
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "entries": [
                {
                    "song_id": "song-a",
                    "source_ref": "train:0",
                    "tja_sha256": "a" * 64,
                    "normalized_title": "same",
                    "primary_split": "train",
                },
                {
                    "song_id": "song-b",
                    "source_ref": "test:0",
                    "tja_sha256": "b" * 64,
                    "normalized_title": "same",
                    "primary_split": "test",
                },
            ],
            "excluded_exact_duplicates": [],
        }
        with self.assertRaisesRegex(ManifestError, "cross primary splits"):
            validate_dataset_manifest(manifest)

    def test_manifest_rejects_duplicate_tja_identity(self) -> None:
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "entries": [
                {
                    "song_id": "song-a",
                    "source_ref": "train:0",
                    "tja_sha256": "a" * 64,
                    "normalized_title": "first",
                    "primary_split": "train",
                },
                {
                    "song_id": "song-b",
                    "source_ref": "train:1",
                    "tja_sha256": "a" * 64,
                    "normalized_title": "second",
                    "primary_split": "validation",
                },
            ],
            "excluded_exact_duplicates": [],
        }
        with self.assertRaisesRegex(ManifestError, "duplicate tja_sha256"):
            validate_dataset_manifest(copy.deepcopy(manifest))

    def test_research_source_snapshot_changes_with_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "TaikoChartEstimator"
            package.mkdir()
            source = package / "example.py"
            source.write_text("value = 1\n", encoding="utf-8")
            first = research_source_snapshot(root)
            source.write_text("value = 2\n", encoding="utf-8")
            second = research_source_snapshot(root)
            self.assertNotEqual(
                first["aggregate_sha256"], second["aggregate_sha256"]
            )


if __name__ == "__main__":
    unittest.main()
