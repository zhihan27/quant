"""验证滚动折缓存的完整性、原子恢复和清理策略。"""

import tempfile
import time
import unittest
import os
from pathlib import Path

import numpy as np

from services.validation_cache import ValidationCache, build_cache_key


class ValidationCacheTests(unittest.TestCase):
    def test_keys_are_stable_and_configuration_sensitive(self):
        identity = {"model": "LightGBM", "horizon": 5, "input_fingerprint": "a" * 64}
        self.assertEqual(build_cache_key(identity), build_cache_key(dict(reversed(list(identity.items())))))
        self.assertNotEqual(build_cache_key(identity), build_cache_key({**identity, "horizon": 20}))
        self.assertNotEqual(build_cache_key(identity), build_cache_key({**identity, "input_fingerprint": "b" * 64}))

    def test_atomic_record_round_trips_predictions_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = ValidationCache(Path(directory), max_bytes=1_000_000, ttl_days=30)
            predictions = np.array([0.1, -0.2, 0.3])
            self.assertTrue(cache.save("a" * 64, predictions, {"baseline": 0.02, "label_processing": {"enabled": False}}))

            restored = cache.load("a" * 64, expected_count=3)

            np.testing.assert_array_equal(restored["predictions"], predictions)
            self.assertEqual(restored["metadata"]["baseline"], 0.02)

    def test_corrupt_and_incomplete_files_are_treated_as_misses(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = ValidationCache(root, max_bytes=1_000_000, ttl_days=30)
            key = "b" * 64
            cache.save(key, np.array([1.0, 2.0]), {"baseline": 1.5, "label_processing": {}})
            cache._path(key).write_bytes(b"corrupted")
            self.assertIsNone(cache.load(key, expected_count=2))
            self.assertFalse(cache._path(key).exists())

            (root / f".{key}.123.interrupted.tmp").write_bytes(b"partial")
            self.assertIsNone(cache.load(key, expected_count=2))

    def test_expired_entries_are_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = ValidationCache(root, max_bytes=1_000_000, ttl_days=1)
            key = "c" * 64
            cache.save(key, np.array([0.5, 0.7]), {"baseline": 0.6, "label_processing": {}})
            expired = time.time() - 2 * 24 * 60 * 60
            cache._path(key).touch()
            os.utime(cache._path(key), (expired, expired))

            cache.cleanup()

            self.assertFalse(cache._path(key).exists())

    def test_expired_entry_is_a_miss_before_access_time_refresh(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = ValidationCache(Path(directory), max_bytes=1_000_000, ttl_days=1)
            key = "d" * 64
            cache.save(key, np.array([0.5, 0.7]), {"baseline": 0.6, "label_processing": {}})
            expired = time.time() - 2 * 24 * 60 * 60
            os.utime(cache._path(key), (expired, expired))

            self.assertIsNone(cache.load(key, expected_count=2))
            self.assertFalse(cache._path(key).exists())


if __name__ == "__main__":
    unittest.main()
