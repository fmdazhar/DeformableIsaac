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
        self._baseline_weights[inds] = value

    def __init__(self, seed=None, **key_ranges):
        # print("SET RNG")
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random

        self.cfg = cfg = {}
        self.indices = indices = {}
        for key, v_range in key_ranges.items():
            bin_size = (v_range[1] - v_range[0]) / v_range[2]
            cfg[key] = np.linspace(v_range[0] + bin_size / 2, v_range[1] - bin_size / 2, v_range[2])
            indices[key] = np.linspace(0, v_range[2]-1, v_range[2])

        self.lows = np.array([range[0] for range in key_ranges.values()])
        self.highs = np.array([range[1] for range in key_ranges.values()])

        self.bin_sizes = {key: (v_range[1] - v_range[0]) / v_range[2] for key, v_range in key_ranges.items()}

        self._raw_grid = np.stack(np.meshgrid(*cfg.values(), indexing='ij'))
        self._idx_grid = np.stack(np.meshgrid(*indices.values(), indexing='ij'))
        self.keys = [*key_ranges.keys()]
        self.grid = self._raw_grid.reshape([len(self.keys), -1])
        self.idx_grid = self._idx_grid.reshape([len(self.keys), -1])

        self._l = l = len(self.grid[0])
        self.ls = {key: len(self.cfg[key]) for key in self.cfg.keys()}

        self.weights = np.zeros(l)
        self._baseline_weights = np.zeros(l)
        self.indices = np.arange(l)

    def __len__(self):
        return self._l

    def __getitem__(self, *keys):
        pass

    def update(self, **kwargs):
        pass

    def sample_bins(self, batch_size, low=None, high=None):
        """default to uniform"""
        if low is not None and high is not None: 
            valid_inds = np.logical_and(
                self.grid >= low[:, None],
                self.grid <= high[:, None]
            ).all(axis=0)
            temp_weights = np.zeros_like(self.weights)
            temp_weights[valid_inds] = self.weights[valid_inds]
            inds = self.rng.choice(self.indices, batch_size, p=temp_weights / temp_weights.sum())
        else:
            inds = self.rng.choice(self.indices, batch_size, p=self.weights / self.weights.sum())

        return self.grid.T[inds], inds

    def sample_uniform_from_cell(self, centroids):
        bin_sizes = np.array([*self.bin_sizes.values()])
        low, high = centroids - bin_sizes / 2, centroids + bin_sizes / 2
        rand = self.rng.uniform(low, high)

        return rand

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

    def update(self, bin_inds, delta, local_range=0.5):
        
        if bin_inds.size == 0:
            return
        new_vals = np.clip(
            self.weights[bin_inds] + delta,
            # floor = baseline if decaying else 0
            self._baseline_weights[bin_inds] if delta < 0 else 0.0,
            1.0
        )
        self.weights[bin_inds] = new_vals

        if delta > 0:
            full_bin_inds = bin_inds[new_vals == 1.0]
            if full_bin_inds.size == 0:
                return
            adjacents = self.get_local_bins(full_bin_inds, ranges=local_range)
            for adjacent in adjacents:
                adjacent_inds = np.array(adjacent.nonzero()[0])
                self.weights[adjacent_inds] = np.clip(self.weights[adjacent_inds] + delta, 0, 1)

if __name__ == '__main__':
    curr = RewardThresholdCurriculum(0,
                                     x=(-3,3,21),
                                     y=(-1,1,7),
                                     z=(-3,3,21))

    BATCH = 1000
    EPOCHS = 1
    thresholds = [0.5, 0.5, 0.5]

    curr.weights[:] = 0.0

    low  = np.array([-1, -1, -1])
    high = np.array([ 1,  1,  1])

    curr.set_to(low, high, value=1.0)

    # Track mean weights over epochs
    mean_weights = []
    
    
    for ep in range(1, EPOCHS+1):     
        rewards = [0,0,0]
        # Update curriculum
        samples, bins = curr.sample(BATCH)
        # curr.update(bins, rewards, thresholds, local_range=0.3)
        mean_weights.append(curr.weights.mean())

    samples, bins = curr.sample(100)
    # unique_bins = np.unique(bins)
    # print("Number of unique bin ids sampled:", len(unique_bins))

    # print("Sampled bins:", bins)
    # plt.scatter(samples[:, 0], samples[:, 2])
    # plt.title('Sampled Points')
    # plt.show()

    # # Plot mean weight progression
    # plt.plot(range(1, EPOCHS+1), mean_weights)
    # plt.xlabel('Epoch')
    # plt.ylabel('Mean Weight')
    # plt.title('Curriculum Weight Growth Over Time')
    # plt.show()

    # x_coords = curr.grid[0]
    # z_coords = curr.grid[2]
    # # Plot the x–z grid
    # plt.figure(figsize=(6,6))
    # plt.scatter(x_coords, z_coords, s=20, alpha=0.7)
    # plt.xlabel('x')
    # plt.ylabel('z')
    # plt.title('Centroid Grid: x vs. z')
    # plt.grid(True)
    # plt.axis('equal')
    # plt.show()