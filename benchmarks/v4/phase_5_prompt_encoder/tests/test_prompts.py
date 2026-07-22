import unittest

import numpy as np

from benchmarks.v4.phase_5_prompt_encoder.src.prompts import (
    dominant_parent_slots,
    hard_region_adjacency,
    sample_connected_region_set,
    sample_region_prompt_episode,
)


class PromptEpisodeTest(unittest.TestCase):
    def setUp(self):
        self.labels = np.asarray([0, 0, 1, 1, 2, 255], dtype=np.int16)
        self.active = np.asarray([1, 1, 1, 1, 1, 0], dtype=bool)
        self.cx = np.asarray([0, 2, 8, 9, 5, 4], dtype=np.float32)
        self.cy = np.asarray([0, 2, 8, 9, 5, 4], dtype=np.float32)

    def sample(self, seed=7):
        return sample_region_prompt_episode(
            self.labels,
            self.active,
            self.cx,
            self.cy,
            (0, 0, 10, 10),
            np.random.default_rng(seed),
            target_class=1,
            min_positive=1,
            max_positive=2,
            min_negative=1,
            max_negative=2,
        )

    def test_prompts_match_target_and_binary_labels(self):
        episode = self.sample()
        self.assertTrue(np.all(self.labels[episode.positive_slots] == 1))
        self.assertTrue(np.all(self.labels[episode.negative_slots] != 1))
        self.assertEqual(episode.binary_region_target.tolist(), [0, 0, 1, 1, 0, 255])
        self.assertTrue(np.all((episode.positive_xy >= 0) & (episode.positive_xy <= 1)))

    def test_sampling_is_deterministic(self):
        one = self.sample(19)
        two = self.sample(19)
        np.testing.assert_array_equal(one.positive_slots, two.positive_slots)
        np.testing.assert_array_equal(one.negative_slots, two.negative_slots)

    def test_purity_mask_is_respected(self):
        eligible = np.asarray([1, 1, 1, 0, 1, 0], dtype=bool)
        episode = sample_region_prompt_episode(
            self.labels,
            self.active,
            self.cx,
            self.cy,
            (0, 0, 10, 10),
            np.random.default_rng(3),
            target_class=1,
            min_positive=1,
            max_positive=1,
            eligible_positive=eligible,
        )
        self.assertEqual(episode.positive_slots.tolist(), [2])

    def test_connected_region_set_and_area_bounds(self):
        hard = np.asarray(
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [2, 2, 3, 3],
                [2, 2, 3, 3],
            ]
        )
        adjacency = hard_region_adjacency(hard, 4)
        slots = sample_connected_region_set(
            adjacency,
            np.ones(4, dtype=bool),
            np.full(4, 4),
            np.random.default_rng(5),
            min_slots=2,
            max_slots=2,
            min_fraction=0.49,
            max_fraction=0.51,
            patch_pixels=16,
        )
        self.assertEqual(len(slots), 2)
        self.assertTrue(adjacency[slots[0], slots[1]])

    def test_dominant_parent_projection(self):
        edge_index = np.asarray([[2, 1], [2, 3], [4, 3]], dtype=np.int16)
        edge_weight = np.asarray([[0.8, 0.2], [0.6, 0.4], [0.1, 0.9]], dtype=np.float32)
        parents = dominant_parent_slots(np.asarray([0, 1, 2]), edge_index, edge_weight)
        self.assertEqual(parents.tolist(), [2, 3])


if __name__ == "__main__":
    unittest.main()
