# enhance AI4Boudnaries by filtering flawed fields & tiles #

import numpy as np
import os
import pandas as pd
import shutil
import xarray as xr
from skimage.segmentation import find_boundaries
from tqdm import tqdm

# define parameter combinations for filtering
area = 25
width = 1
solidity = 0.4
euler = 5
landcov = 25
agri_prop = 0.5
field_prop = 0.1
n_fields = 5

# define dirs
data_dir = "d:/thesis/data"
eda_dir = "d:/thesis/results/eda/"
save_dir = os.path.join(data_dir, "ai4boundaries", "filtered")
os.makedirs(save_dir, exist_ok=True)

# filter fields & tiles according to defined criteria
# read summary on labels (-> eda/eda_labels.py)
stats_fields = pd.read_csv(
    os.path.join(eda_dir, "original_stats_fields.csv"), index_col=0, header=[0, 1]
)
stats_fields = stats_fields.drop(
    columns=[x for x in stats_fields.columns if x[0] == "clc"]
)
stats_fields.columns = stats_fields.columns.get_level_values(1)
stats_fields = stats_fields.rename(columns={stats_fields.columns[0]: "field_id"})
# apply shape-related & clc+ exclusion criteria
fields_clean = stats_fields[
    (stats_fields["area"] > area)
    & (stats_fields["axis_minor_length"] > width)
    & (stats_fields["solidity"] > solidity)
    & (stats_fields["euler_8"] <= euler)
    & (stats_fields[[str(x) for x in [1, 10]]].sum(axis=1) <= landcov)
]
# re-read summary on labels & apply tile-based clc exclusion criteria
stats_fields = pd.read_csv(
    os.path.join(eda_dir, "original_stats_fields.csv"), index_col=0, header=[0, 1]
)
stats_fields = stats_fields.drop(
    columns=[x for x in stats_fields.columns if x[0] == "clc_plus"]
)
stats_fields = stats_fields.drop(
    columns=[x for x in stats_fields.columns if x[0] == "obj_stats"]
)
stats_fields.columns = stats_fields.columns.get_level_values(1)
stats_fields = stats_fields.rename(columns={stats_fields.columns[0]: "field_id"})
fields_clean = pd.merge(
    fields_clean.iloc[:, :10], stats_fields, on=["tile_id", "field_id"]
)
agri_classes = np.arange(12, 23)
agri_per_tile = (
    fields_clean.groupby("tile_id").sum()[[str(x) for x in agri_classes]].sum(axis=1)
)
fields_per_tile = fields_clean.groupby("tile_id").sum()["area"]
bool_prop_agri = (agri_per_tile / fields_per_tile) >= agri_prop
index_prop_agri = fields_clean.groupby("tile_id").sum()[bool_prop_agri].index
fields_clean = fields_clean[fields_clean.index.isin(index_prop_agri)]
# apply filtering based on minimum fraction of fields/tile & absolute number of fields/tile
bool_prop_thres = (
    fields_clean.groupby("tile_id").sum()["area"] / (256 * 256)
) >= field_prop
index_prop_thres = fields_clean.groupby("tile_id").sum()[bool_prop_thres].index
fields_clean = fields_clean[fields_clean.index.isin(index_prop_thres)]
bool_n_thres = fields_clean.groupby("tile_id").size() >= n_fields
index_n_thres = fields_clean.groupby("tile_id").size()[bool_n_thres].index
fields_clean = fields_clean[fields_clean.index.isin(index_n_thres)]

# write filtered tiles to disk
tile_set = np.unique(fields_clean.index)
for i, tile in tqdm(enumerate(tile_set), total=len(tile_set)):
    # read label data
    tile_path = f"{tile}_S2_10m_256.nc"
    tile_path = os.path.join(data_dir, "ai4boundaries", "original", tile_path)
    label_path = tile_path.replace("_S2_10m_256.nc", "_S2label_10m_256.tif")
    label_ds = xr.open_dataset(label_path)
    proj_wkt = label_ds["spatial_ref"].attrs["spatial_ref"]
    label_ds.rio.write_crs(proj_wkt, inplace=True)
    # get single bands of label ds
    ext = np.array(label_ds["band_data"][0])
    bound = np.array(label_ds["band_data"][1])
    dist = np.array(label_ds["band_data"][2])
    enum = np.array(label_ds["band_data"][3])
    # evaluate filter
    good_ids = fields_clean[fields_clean.index == tile]["field_id"]
    na_idxs = np.where(np.isnan(enum), True, False)
    bad_idxs = ~np.isin(enum, good_ids)
    # create new results
    new_enum = np.where(np.logical_and(~na_idxs, ~bad_idxs), enum, np.nan)
    new_dist = np.where(np.logical_and(~na_idxs, ~bad_idxs), dist, 0)
    new_bound = find_boundaries(np.nan_to_num(new_enum))
    new_ext = np.where(np.logical_and(~na_idxs, ~bad_idxs), ext, 0)
    # modify original label ds
    for i, arr in enumerate([new_ext, new_bound, new_dist, new_enum]):
        label_ds["band_data"][i] = arr
    # write to label ds to disk
    dst = os.path.join(save_dir, f"{tile}_S2label_10m_256.tif")
    label_ds["band_data"].rio.to_raster(dst, compress="LZW")
    # copy satellite arr from original src
    src = tile_path
    dst = os.path.join(save_dir, f"{tile}_S2_10m_256.nc")
    shutil.copy(src, dst)
