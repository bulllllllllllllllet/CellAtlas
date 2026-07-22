import unittest

from benchmarks.v4.phase_1_multiscale.src.dataset import BalancedPatchSampler


class _Dataset:
    def __init__(self):
        self.rows = [{"sampling_group": group} for group in ("a", "b") for _ in range(20)]

    def __len__(self):
        return len(self.rows)


class DDPSamplerTest(unittest.TestCase):
    def test_ranks_are_disjoint_and_epoch_is_deterministic(self):
        dataset = _Dataset(); ratios = {"a": 0.5, "b": 0.5}
        samplers = [BalancedPatchSampler(dataset, ratios, seed=7, rank=rank, world_size=4) for rank in range(4)]
        epoch0 = [list(sampler) for sampler in samplers]
        self.assertEqual([len(ids) for ids in epoch0], [10, 10, 10, 10])
        single_rank = list(BalancedPatchSampler(dataset, ratios, seed=7))
        self.assertEqual([item for batch in zip(*epoch0) for item in batch], single_rank)
        self.assertEqual(epoch0, [list(sampler) for sampler in samplers])
        for sampler in samplers: sampler.set_epoch(1)
        self.assertNotEqual(epoch0, [list(sampler) for sampler in samplers])

    def test_fixed_global_epoch_budget(self):
        dataset = _Dataset(); ratios = {"a": 0.5, "b": 0.5}
        samplers = [BalancedPatchSampler(dataset, ratios, seed=7, rank=rank, world_size=4, epoch_size=24) for rank in range(4)]
        self.assertEqual([len(sampler) for sampler in samplers], [6, 6, 6, 6])


if __name__ == "__main__":
    unittest.main()
