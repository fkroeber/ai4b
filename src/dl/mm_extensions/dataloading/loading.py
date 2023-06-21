import numpy as np
import xarray as xr
from mmcv.transforms.base import BaseTransform
from mmcv.transforms.builder import TRANSFORMS
from typing import Optional


@TRANSFORMS.register_module(force=True)
class LoadAi4b(BaseTransform):
    """Load an image from file.

    Required Keys:

    - img_path

    Modified Keys:

    - img
    - img_shape
    - ori_shape
    """

    def __init__(self) -> None:
        pass

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results["img_path"]

        # read satellite imagery
        tile_ds = xr.open_dataset(filename)
        proj_wkt = tile_ds["spatial_ref"].attrs["spatial_ref"]
        tile_ds.rio.write_crs(proj_wkt, inplace=True)

        # modify satellite imagery by rescaling & clipping optical bands to [0,1]
        tile_arr = np.array(tile_ds.to_array()).reshape(-1, 256, 256)
        tile_arr[:24, ...] = tile_arr[:24, ...] / 10000

        # rescale NDVI to [0,1]
        tile_arr[:24, ...] = np.where(tile_arr[:24, ...] > 1, 1, tile_arr[:24, ...])
        tile_arr[:24, ...] = tile_arr[:24, ...] * 0.5 + 0.5

        # output all bands being scaled between [0,255] in int8 format
        img = np.nan_to_num(tile_arr)
        img = 255 * np.transpose(img, (1, 2, 0))
        img = img.astype(np.uint8)

        results["img"] = img
        results["img_shape"] = img.shape[:2]
        results["ori_shape"] = img.shape[:2]
        return results
