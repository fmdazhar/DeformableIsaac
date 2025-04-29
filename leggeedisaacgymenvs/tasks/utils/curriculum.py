import numpy as np
from matplotlib import pyplot as plt


class Curriculum:
    def set_to(self, low, high, value=1.0):
        inds = np.logical_and(
            self.grid >= low[:, None],
            self.grid <= high[:, None]
        ).all(axis=0)

        assert len(inds) != 0, "You are intializing your distribution with an empty domain!"

        self.weights[inds] = value

    def __init__(self, seed, **key_ranges):
        # print("SET RNG")
        self.rng = np.random.RandomState(seed)

        self.cfg = cfg = {}
        self.indices = indices = {}
        for key, v_range in key_ranges.items():
            bin_size = (v_range[1] - v_range[0]) / v_range[2]
            cfg[key] = np.linspace(v_range[0] + bin_size / 2, v_range[1] - bin_size / 2, v_range[2])
            indices[key] = np.linspace(0, v_range[2]-1, v_range[2])

        self.lows = np.array([range[0] for range in key_ranges.values()])
        self.highs = np.array([range[1] for range in key_ranges.values()])

        # self.bin_sizes = {key: arr[1] - arr[0] for key, arr in cfg.items()}
        self.bin_sizes = {key: (v_range[1] - v_range[0]) / v_range[2] for key, v_range in key_ranges.items()}

        self._raw_grid = np.stack(np.meshgrid(*cfg.values(), indexing='ij'))
        self._idx_grid = np.stack(np.meshgrid(*indices.values(), indexing='ij'))
        self.keys = [*key_ranges.keys()]
        self.grid = self._raw_grid.reshape([len(self.keys), -1])
        self.idx_grid = self._idx_grid.reshape([len(self.keys), -1])
        # self.grid = np.stack([params.flatten() for params in raw_grid])

        self._l = l = len(self.grid[0])
        self.ls = {key: len(self.cfg[key]) for key in self.cfg.keys()}

        self.weights = np.zeros(l)
        self.indices = np.arange(l)

    def __len__(self):
        return self._l

    def __getitem__(self, *keys):
        pass

    def update(self, **kwargs):
        # bump the envelop if
        pass

    def sample_bins(self, batch_size, low=None, high=None):
        """default to uniform"""
        if low is not None and high is not None: # if bounds given
            valid_inds = np.logical_and(
                self.grid >= low[:, None],
                self.grid <= high[:, None]
            ).all(axis=0)
            temp_weights = np.zeros_like(self.weights)
            temp_weights[valid_inds] = self.weights[valid_inds]
            inds = self.rng.choice(self.indices, batch_size, p=temp_weights / temp_weights.sum())
        else: # if no bounds given
            # print(self.weights, self.weights.sum())
            inds = self.rng.choice(self.indices, batch_size, p=self.weights / self.weights.sum())

        return self.grid.T[inds], inds

    def sample_uniform_from_cell(self, centroids):
        bin_sizes = np.array([*self.bin_sizes.values()])
        low, high = centroids + bin_sizes / 2, centroids - bin_sizes / 2
        rand = self.rng.uniform(low, high)

        return rand#.clip(self.lows, self.highs)

    def sample(self, batch_size, low=None, high=None):
        cgf_centroid, inds = self.sample_bins(batch_size, low=low, high=high)
        return np.stack([self.sample_uniform_from_cell(v_range) for v_range in cgf_centroid]), inds


class RewardThresholdCurriculum(Curriculum):
    def __init__(self, seed, **kwargs):
        super().__init__(seed, **kwargs)

    def get_local_bins(self, bin_inds, ranges=0.1):
        if isinstance(ranges, float):
            ranges = np.ones(self.grid.shape[0]) * ranges
        bin_inds = bin_inds.reshape(-1)

        adjacent_inds = np.logical_and(
            self.grid[:, None, :].repeat(bin_inds.shape[0], axis=1) >= self.grid[:, bin_inds, None] - ranges.reshape(-1, 1, 1),
            self.grid[:, None, :].repeat(bin_inds.shape[0], axis=1) <= self.grid[:, bin_inds, None] + ranges.reshape(-1, 1, 1)
        ).all(axis=0)

        return adjacent_inds

    def update(self, bin_inds, task_rewards, success_thresholds, local_range=0.5):

        is_success = 1.
        for task_reward, success_threshold in zip(task_rewards, success_thresholds):
            is_success = is_success * (task_reward > success_threshold).cpu()
        if len(success_thresholds) == 0:
            is_success = np.array([False] * len(bin_inds))
        else:
            is_success = np.array(is_success.bool())

        self.weights[bin_inds[is_success]] = np.clip(self.weights[bin_inds[is_success]] + 0.2, 0, 1)
        adjacents = self.get_local_bins(bin_inds[is_success], ranges=local_range)
        for adjacent in adjacents:
            adjacent_inds = np.array(adjacent.nonzero()[0])
            self.weights[adjacent_inds] = np.clip(self.weights[adjacent_inds] + 0.2, 0, 1)

