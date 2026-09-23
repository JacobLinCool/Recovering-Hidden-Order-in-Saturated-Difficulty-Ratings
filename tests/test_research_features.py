from __future__ import annotations

import unittest

from TaikoChartEstimator.research.features import (
    BASE_FEATURE_NAMES,
    extract_global_features,
)


class ResearchFeaturesTest(unittest.TestCase):
    def test_extracts_finite_features_and_excludes_end_marker(self) -> None:
        chart = {
            "segments": [
                {
                    "timestamp": 0.0,
                    "measure_num": 4,
                    "measure_den": 4,
                    "notes": [
                        {
                            "timestamp": 0.0,
                            "note_type": "Don",
                            "bpm": 120.0,
                            "scroll": 1.0,
                            "delay": 0.0,
                        },
                        {
                            "timestamp": 0.5,
                            "note_type": "DonBig",
                            "bpm": 120.0,
                            "scroll": -2.0,
                            "delay": 0.0,
                        },
                        {
                            "timestamp": 1.0,
                            "note_type": "EndOf",
                            "bpm": 120.0,
                            "scroll": 1.0,
                            "delay": 0.0,
                        },
                    ],
                }
            ]
        }
        features = extract_global_features(chart)
        self.assertEqual(tuple(features), BASE_FEATURE_NAMES)
        self.assertEqual(features["note_count"], 2.0)
        self.assertEqual(features["chart_duration"], 0.5)
        self.assertEqual(features["large_note_fraction"], 0.5)
        self.assertEqual(features["abs_scroll_max"], 2.0)


if __name__ == "__main__":
    unittest.main()
