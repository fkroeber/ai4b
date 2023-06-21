import geopandas as gpd
import numpy as np
from rasterio import features
from shapely.geometry import shape


def rs_to_gdf(rs_raster):
    """
    Vectorise an input raster
    * rasterio: rs.features.shapes <-> rs.features.rasterise
    * rioxarray: geocube vectorise (req python >= 3.9) <-> rasterise
    """
    # create intial gdf with polygons
    shps = features.shapes(rs_raster.astype(np.float32))
    shps = [(shape(shp), val) for shp, val in shps if ~np.isnan(val)]
    gdf = gpd.GeoDataFrame(
        {"shp_idx": [x[1] for x in shps], "geometry": [x[0] for x in shps]}
    )
    # create multipolygon gdf out of shattered polygons
    merged_polygons = []
    grouped = gdf.groupby("shp_idx")
    for val, group in grouped:
        merged_polygon = group.geometry.unary_union
        merged_polygons.append(merged_polygon)
    merged_gdf = gpd.GeoDataFrame(
        {"shp_idx": grouped.groups.keys(), "geometry": merged_polygons}
    )
    return merged_gdf
