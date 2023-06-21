# get basic descriptive statistics for labels on tile & individual field level #

import geopandas as gpd
import numpy as np
import pandas as pd
import os
import xarray as xr
import rioxarray as rxr

from geocube.api.core import make_geocube
from pyproj import Transformer
from skimage.measure import euler_number, regionprops_table
from tqdm import tqdm
from xrspatial.zonal import crosstab

# define dirs
data_dir = "d:/thesis/data"
save_dir = "d:/thesis/results/eda"
os.makedirs(save_dir, exist_ok=True)

# define version of ai4b to be used
ai4b_v = "filtered"
ai4b_dir = os.path.join(data_dir, "ai4boundaries", ai4b_v)

# helper functions for object properties
def euler_4(regionmask):
    return euler_number(regionmask, connectivity=1)


def euler_8(regionmask):
    return euler_number(regionmask, connectivity=2)


# read CLC information & reproject it to same CRS as AI4coundaries
clc_data_path = os.path.join(
    data_dir, "clc2018_raster100m", "DATA", "U2018_CLC2018_V2020_20u1.tif"
)
clc = rxr.open_rasterio(clc_data_path)
clc = clc.rio.reproject("EPSG:3035")

# read CLC+ (already in EPSG3035)
clc_plus_data_path = os.path.join(
    data_dir,
    "clcplus2018_raster10m",
    "DATA",
    "CLMS_CLCplus_RASTER_2018_010m_eu_03035_V1_1.tif",
)
clc_plus = rxr.open_rasterio(clc_plus_data_path)

# define value ranges for clc
# NaN: 44 (CLC), 254 & 255 (CLC_Plus)
clc_vals = np.arange(1, 45)
clc_plus_vals = [*np.arange(1, 12), 254, 255]

# biogeographical regions loading
biogeo_regions = gpd.read_file(
    os.path.join(data_dir, "biogeoregions", "BiogeoRegions2016.shp")
)
regions_cat = biogeo_regions["short_name"].to_dict()
keys = list(regions_cat.keys())
vals = list(regions_cat.values())
biogeo_regions["short_num"] = [
    keys[vals.index(x)] for x in biogeo_regions["short_name"]
]
biogeo = make_geocube(
    biogeo_regions,
    measurements=["short_num"],
    resolution=(-1000, 1000),
)

# ERA5 data loading
era_ds = xr.open_dataset(os.path.join(data_dir, "era5", "monthly_prec_temp.nc"))
era_5yavg = era_ds.mean(dim="time")
# crs re-projection to epsg:3035
era_5yavg.rio.write_crs("EPSG:4326", inplace=True)
era_5yavg = era_5yavg.rio.reproject("EPSG:3035")
# converting temp & precipitation to common units
era_5yavg["t2m"] = era_5yavg["t2m"] - 273.15
era_5yavg["tp"] = 365 * era_5yavg["tp"] * 1000

# read file with overview on tiles
tile_df = pd.read_csv(
    os.path.join(
        data_dir, "ai4boundaries", "ai4boundaries_ftp_urls_sentinel2_split.csv"
    )
)
tile_df.insert(1, "country", [x.split("_")[0] for x in tile_df["file_id"]])
tile_df.dropna(inplace=True)

stats_tiles = []
stats_fields = []

for _, row in tqdm(tile_df.iterrows(), total=tile_df.shape[0], smoothing=0.01):
    # compose tile & label paths
    tile_path = f"{row['file_id']}_S2_10m_256.nc"
    tile_path = os.path.join(ai4b_dir, tile_path)
    label_path = tile_path.replace("_S2_10m_256.nc", "_S2label_10m_256.tif")

    # read data cubes
    # skip if file not found or spatial ref not defined
    try:
        # for tile array
        tile_ds = xr.open_dataset(tile_path)
        proj_wkt = tile_ds["spatial_ref"].attrs["spatial_ref"]
        tile_ds.rio.write_crs(proj_wkt, inplace=True)
        assert str(tile_ds.rio.crs) == "EPSG:3035"
        # for label array
        label_ds = xr.open_dataset(label_path)
        proj_wkt = label_ds["spatial_ref"].attrs["spatial_ref"]
        label_ds.rio.write_crs(proj_wkt, inplace=True)
        assert str(label_ds.rio.crs) == "EPSG:3035"
    except:
        continue

    # extract arrays for fields labels
    field_enum = label_ds["band_data"][3]
    field_enum_np = np.array(field_enum)
    extent_mask_np = np.array(label_ds["band_data"][0])

    # get position metadata
    meta = {}
    bbox = label_ds.rio.bounds()
    x = (bbox[1] + bbox[3]) / 2
    y = (bbox[0] + bbox[2]) / 2
    transformer = Transformer.from_crs("EPSG:3035", "EPSG:4326")
    meta["lat"], meta["lon"] = transformer.transform(x, y)

    # get biogeo regions
    try:
        bio_geo_idx = int(biogeo.sel(x=y, y=x, method="nearest")["short_num"])
    # use nearest neighbour window as locations close to ocean gets NaN assigned otherwise
    except ValueError:
        bbox_bounds = np.array(bbox) + [-2500, -2500, 2500, 2500]
        biogeo_clip = np.array(biogeo.rio.clip_box(*bbox_bounds).to_array()).flatten()
        biogeo_clip = biogeo_clip[~np.isnan(biogeo_clip)]
        vals, counts = np.unique(biogeo_clip, return_counts=True)
        bio_geo_idx = int(vals[np.argmax(counts)])
    meta["biogeo_region"] = regions_cat[bio_geo_idx]

    # get era meteorological data
    meta["temp"] = float(era_5yavg.sel(x=y, y=x, method="nearest")["t2m"])
    meta["prec"] = float(era_5yavg.sel(x=y, y=x, method="nearest")["tp"])

    # calculation of overall tile stats
    tile_stats = {}
    tile_stats["n_fields"] = len(np.unique(field_enum_np)) - 1
    tile_stats["prop_fields"] = extent_mask_np.sum() / extent_mask_np.size
    if tile_stats["n_fields"] != 0:
        tile_stats["avg_size_ha"] = np.round(
            extent_mask_np.sum() / (tile_stats["n_fields"] * 100), 2
        )
    else:
        tile_stats["avg_size_ha"] = np.nan

    # in case of no fields only write tile stats
    if tile_stats["n_fields"] == 0:
        meta = pd.DataFrame(meta, index=np.arange(1))
        meta_cols = pd.MultiIndex.from_product([["meta"], meta.columns])
        meta.columns = meta_cols
        tile_stats = pd.DataFrame(tile_stats, index=np.arange(1))
        tile_stats_cols = pd.MultiIndex.from_product([["stats"], tile_stats.columns])
        tile_stats.columns = tile_stats_cols
        tile_stats = pd.merge(meta, tile_stats, left_index=True, right_index=True)
        tile_stats.insert(0, "tile_id", row["file_id"])
        stats_tiles.append(tile_stats)

    else:
        # clc landcover related stats
        bbox_bounds = np.array(bbox) + [-250, -250, 250, 250]
        clc_cropped = clc.rio.clip_box(*bbox_bounds)
        clc_cropped = clc_cropped.interp_like(field_enum, method="nearest")

        clc_per_field = crosstab(field_enum, clc_cropped[0])
        clc_per_tile = clc_per_field.sum(axis=0)

        # clc+ landcover related stats
        bbox_bounds = np.array(bbox) + [-100, -100, 100, 100]
        clc_plus_cropped = clc_plus.rio.clip_box(*bbox_bounds)
        clc_plus_cropped = clc_plus_cropped.interp_like(field_enum, method="nearest")

        clc_plus_per_field = crosstab(field_enum, clc_plus_cropped[0])
        clc_plus_per_tile = clc_plus_per_field.sum(axis=0)

        # calculate field obj stats
        props = regionprops_table(
            field_enum_np.astype(np.int16),
            properties=(
                "label",
                "area",
                "perimeter",
                "solidity",
                "axis_major_length",
                "axis_minor_length",
                "eccentricity",
            ),
            extra_properties=(euler_4, euler_8),
        )
        props = pd.DataFrame(props)
        props["circularity"] = (4 * np.pi * props["area"]) / np.power(
            props["perimeter"], 2
        )

        # merge all stats on tile level into one df
        meta = pd.DataFrame(meta, index=np.arange(1))
        meta_cols = pd.MultiIndex.from_product([["meta"], meta.columns])
        meta.columns = meta_cols

        tile_stats = pd.DataFrame(tile_stats, index=np.arange(1))
        tile_stats_cols = pd.MultiIndex.from_product([["stats"], tile_stats.columns])
        tile_stats.columns = tile_stats_cols

        clc_tile = pd.DataFrame(clc_per_tile, index=clc_vals).T
        clc_tile_cols = pd.MultiIndex.from_product([["clc"], clc_tile.columns])
        clc_tile.columns = clc_tile_cols

        clc_plus_tile = pd.DataFrame(clc_plus_per_tile, index=clc_plus_vals).T
        clc_plus_tile_cols = pd.MultiIndex.from_product(
            [["clc_plus"], clc_plus_tile.columns]
        )
        clc_plus_tile.columns = clc_plus_tile_cols

        tile_stats = pd.merge(meta, tile_stats, left_index=True, right_index=True)
        tile_stats = pd.merge(tile_stats, clc_tile, left_index=True, right_index=True)
        tile_stats = pd.merge(
            tile_stats, clc_plus_tile, left_index=True, right_index=True
        )
        tile_stats.insert(0, "tile_id", row["file_id"])

        # merge all stats on fields level into one df
        props = props.set_index("label")
        props_cols = pd.MultiIndex.from_product([["obj_stats"], props.columns])
        props.columns = props_cols

        clc_field = pd.DataFrame(clc_per_field.set_index("zone").T, index=clc_vals).T
        clc_field.index = clc_field.index.astype("int64")
        clc_field.index.names = ["label"]
        clc_field_cols = pd.MultiIndex.from_product([["clc"], clc_field.columns])
        clc_field.columns = clc_field_cols

        clc_plus_field = pd.DataFrame(
            clc_plus_per_field.set_index("zone").T, index=clc_plus_vals
        ).T
        clc_plus_field.index = clc_plus_field.index.astype("int64")
        clc_plus_field.index.names = ["label"]
        clc_plus_field_cols = pd.MultiIndex.from_product(
            [["clc_plus"], clc_plus_field.columns]
        )
        clc_plus_field.columns = clc_plus_field_cols

        field_stats = pd.merge(props, clc_field, left_index=True, right_index=True)
        field_stats = pd.merge(
            field_stats, clc_plus_field, left_index=True, right_index=True
        )
        field_stats.index.names = ["field_id"]
        field_stats = field_stats.reset_index()
        field_stats.insert(0, "tile_id", row["file_id"])

        # write to summarising lists
        stats_fields.append(field_stats)
        stats_tiles.append(tile_stats)

# concat results for tiles
stats_tiles = pd.concat(stats_tiles)
stats_tiles = stats_tiles.set_index(stats_tiles.columns[0])
nan_cols = stats_tiles.iloc[:, 2:].columns
stats_tiles[nan_cols] = stats_tiles[nan_cols].replace({"0": np.nan, 0: np.nan})
stats_tiles.index.names = ["tile_id"]

# add general metadata information from intial tile_df
tile_df = tile_df.set_index("file_id")
tile_df_cols = pd.MultiIndex.from_product([["meta"], tile_df.columns])
tile_df.columns = tile_df_cols
stats_tiles = stats_tiles.join(tile_df).sort_index()

# re-arrange columns with meta info appearing first
meta_cols = [x for x in stats_tiles.columns if "meta" in x]
for meta_col in meta_cols:
    col = stats_tiles.pop(meta_col)
    stats_tiles.insert(0, col.name, col)

# add unique id for each tile
country_code = {"AT": 1, "ES": 2, "FR": 3, "LU": 4, "NL": 5, "SE": 6, "SI": 7}
uids = [
    int(f"{country_code[x]}{int(y.split('_')[-1]):07d}")
    for x, y in zip(stats_tiles[("meta", "country")], stats_tiles.index)
]
stats_tiles.insert(0, ("meta", "uid"), uids)

# concat results for fields
stats_fields = pd.concat(stats_fields)
stats_fields = stats_fields.set_index(stats_fields.columns[0])
nan_cols = [x for x in stats_fields.columns if "clc" in x[0]]
stats_fields[nan_cols] = stats_fields[nan_cols].replace({"0": np.nan, 0: np.nan})
stats_fields.index.names = ["tile_id"]
stats_fields.sort_index(inplace=True)

# write results to disk
stats_tiles.to_csv(os.path.join(save_dir, f"{ai4b_v}_stats_tiles.csv"))
stats_fields.to_csv(os.path.join(save_dir, f"{ai4b_v}_stats_fields.csv"))
