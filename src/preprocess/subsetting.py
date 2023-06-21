# create subsets for assessing method based on different sizes #

import numpy as np
import os
import pandas as pd
from itertools import combinations
from scipy.spatial import cKDTree as KDTree
from tqdm import tqdm
from utils.common import normalise

# subsetting parameters
subset_sizes = [10, 50, 100, 250, 1000, 4000]
n_subsets = 5
n_trials = 25000

# define exploratory data dir
eda_dir = "d:/thesis/results/eda/"

# function for kl calculation
def KLdivergence(x, y):
    """
    Compute the Kullback-Leibler divergence between two multivariate samples.
    src: https://gist.github.com/atabakd/ed0f7581f8510c8587bc2f41a094b518
    """
    # sanity checks
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    n, d = x.shape
    m, dy = y.shape
    assert d == dy
    # build tree representation
    xtree = KDTree(x)
    ytree = KDTree(y)
    # get two nearest neighbours for x
    r = xtree.query(x, k=2, eps=0.01, p=2)[0][:, 1]
    s = ytree.query(x, k=1, eps=0.01, p=2)[0]
    return -np.log(r / s).sum() * d / n + np.log(m / (n - 1.0))


# read summary on labels (-> eda/eda_labels.py)
stats_fields = pd.read_csv(
    os.path.join(eda_dir, "filtered_stats_fields.csv"), index_col=0, header=[0, 1]
)
stats_fields = stats_fields.drop(
    columns=[x for x in stats_fields.columns if x[0] == "clc"]
)
stats_fields.columns = stats_fields.columns.get_level_values(1)
stats_fields = stats_fields.rename(columns={stats_fields.columns[0]: "field_id"})

stats_tiles = pd.read_csv(
    os.path.join(eda_dir, "filtered_stats_tiles.csv"), index_col=0, header=[0, 1]
)
stats_tiles.columns = stats_tiles.columns.get_level_values(1)

# read deep feature description for images (-> eda/eda_images.py)
deep_feats = pd.read_csv(os.path.join(eda_dir, "filtered_tiles_feats.csv"), index_col=0)
feats_df = deep_feats[
    (deep_feats["img_mode"] == "true_color")
    & (deep_feats["net"] == "resnet")
    & (deep_feats["reduce_alg"] == "tsne")
    & (deep_feats["topo_balance"] == 30)
]
feats_df = feats_df[["x1", "x2"]]

# compile additional label information used for stratification of sampling
sampling_dis = feats_df.join(normalise(stats_tiles[["avg_size_ha"]]))
sampling_dis = sampling_dis.join(
    normalise(stats_fields.groupby("tile_id").mean()["solidity"])
)

all_subsets = []
for n_samples in subset_sizes:

    # brute-force assessment of random subsets
    kl_res = []
    for seed in tqdm(np.arange(n_trials), desc=f"Sampling subsets of size {n_samples}"):
        np.random.seed(seed)
        subset = sampling_dis.sample(n_samples)
        X, y = np.array(sampling_dis), np.array(subset)
        noise = np.random.multivariate_normal(
            np.zeros(sampling_dis.shape[1]),
            np.identity(sampling_dis.shape[1]) / 10e5,
            len(y),
        )
        kl_res.append(KLdivergence(X, y + noise))

    # get maximum kl_div samples for sanity check
    kl_div_idxmaxs = np.argsort(kl_res)[-n_subsets:]
    kl_div_maxs = np.array(kl_res)[kl_div_idxmaxs]

    # get samples with minimum kl_div as candidates for subsets
    kl_div_idxmins = np.argsort(kl_res)[: int(max(2 * n_subsets, 0.01 * n_trials))]
    subsets_candidates = []
    for seed_idx in kl_div_idxmins:
        np.random.seed(seed_idx)
        subset_kl_min = feats_df.sample(n_samples)
        subsets_candidates.append(set(subset_kl_min.index))

    # consolidate selection by using sets with minimum overlap to each other
    while len(kl_div_idxmins) > n_subsets - 1:
        # evaluate number of overlapping tiles for pairs of sets
        idxs = list(combinations(np.arange(len(kl_div_idxmins)), 2))
        pairs = combinations(subsets_candidates, 2)
        nt = lambda a, b: a.intersection(b)
        ovlps = [len(nt(*t)) for t in pairs]
        ovlps_dict = dict(zip(idxs, ovlps))
        # create overlapping matrix
        size = max(max(k) for k in ovlps_dict.keys()) + 1
        arr = np.zeros((size, size), dtype=int)
        for (i, j), val in ovlps_dict.items():
            arr[i][j] = val
            arr[j][i] = val
        # breaking condition
        if len(kl_div_idxmins) == n_subsets:
            mean_ovlps = arr.mean(axis=0)
            max_ovlps = arr.max(axis=0)
            break
        # remove set with highest overlap
        kl_div_idxmins = np.delete(kl_div_idxmins, np.argmax(arr.sum(axis=0)))
        subsets_candidates.pop(np.argmax(arr.max(axis=0)))

    # get kl divs for chosen samples
    kl_div_mins = np.array(kl_res)[kl_div_idxmins]

    # for each subset get list of indices & other information
    res_subsets = []
    for i, seed_idx in enumerate(kl_div_idxmins):
        np.random.seed(seed_idx)
        subset = sampling_dis.sample(n_samples)
        res_subsets.append(
            pd.Series(
                {
                    "subset_idx": i,
                    "seed": seed_idx,
                    "n_samples": n_samples,
                    "n_fields": sum(stats_fields.index.isin(subset.index)),
                    "tile_list": list(subset.index),
                    "mean_ovlps": mean_ovlps,
                    "max_ovlps": max_ovlps,
                }
            )
        )
    res_subsets = pd.DataFrame(res_subsets)
    all_subsets.append(res_subsets)

# write resulting subsets to disk
subsets = pd.concat(all_subsets)
subsets.to_csv(os.path.join(eda_dir, "subsets_df.csv"), index=False)
