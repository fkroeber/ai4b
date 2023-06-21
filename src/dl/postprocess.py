import multiprocess
import os
from functools import partial
from tqdm import tqdm


class InstancePostProcessor:
    """
    Post-processing for predictions produced by instance segmentation networks,
    takes as input a prediction dict containing the masks and scores information
    within the pred_instances attribute, outputs a 2d array segmentation array
    """

    def __init__(
        self,
        pred,
        area_thres=25,
        score_thres=0.5,
        iou_thres=0.1,
    ):
        self.pred = pred
        self.area_t = area_thres
        self.score_t = score_thres
        self.iou_t = iou_thres
        self._imports_for_spawned_processes()

    def mask_pp(self):
        """
        Postprocessing based on mask IoUs,
        assumes that inference result obtained via test.py
        was run with very high IoU & low score threshold such that
        filtering can be done in this step
        """
        # get scores & predictions
        scores = self.pred["pred_instances"]["scores"].numpy()
        masks = self.pred["pred_instances"]["masks"]

        # decode single 256x256 map to bitmap
        if torch.is_tensor(masks):
            maskedArr = np.transpose(masks.numpy().astype(np.int8), (1, 2, 0))
        else:
            maskedArr = cocomask.decode(masks)

        # remove segments below area threshold
        bool_area = maskedArr.sum(axis=(0, 1)) > self.area_t
        area_idx = np.arange(len(bool_area))[bool_area]
        maskedArr = maskedArr[..., area_idx]
        scores = scores[area_idx]

        # remove segments below a certain score
        bool_score_idx = scores > self.score_t
        maskedArr = maskedArr[..., bool_score_idx]
        scores = scores[bool_score_idx]

        # convert bitmaps to polygons
        polys = [self._convert_to_poly(x) for x in maskedArr.transpose(2, 0, 1)]

        # create df
        data = {"geometry": polys, "score": scores}
        gdf = gpd.GeoDataFrame(data)

        # intersection with itself
        df_combi = gpd.sjoin(gdf, gdf, how="left", predicate="intersects")
        df_combi.reset_index(names="index_left", inplace=True)
        df_combi = df_combi[["index_left", "index_right", "score_left", "score_right"]]
        df_combi = df_combi[df_combi["index_left"] != df_combi["index_right"]]

        df_combi = pd.merge(
            df_combi, gdf["geometry"], left_on="index_left", right_index=True
        )
        df_combi = pd.merge(
            df_combi, gdf["geometry"], left_on="index_right", right_index=True
        )
        df_combi = df_combi.rename(
            columns={"geometry_x": "geom_left", "geometry_y": "geom_right"}
        )
        df_combi = gpd.GeoDataFrame(df_combi)

        intersection = df_combi["geom_left"].intersection(df_combi["geom_right"])
        union = df_combi["geom_left"].union(df_combi["geom_right"])
        df_combi["iou"] = intersection.area / union.area
        df_combi.sort_values("index_left", inplace=True)

        # iterate through descending scores & remove overlapping preds
        dropped_idxs = []
        for i, r in gdf.sort_values("score", ascending=False).iterrows():
            ovlps = df_combi[df_combi["index_left"] == i]
            ovlps = ovlps[ovlps["score_left"] > ovlps["score_right"]]
            ovlps = ovlps[ovlps["iou"] > self.iou_t]
            if len(ovlps):
                drop_idxs = ovlps["index_right"]
                drop_idxs = drop_idxs[~drop_idxs.isin(dropped_idxs)]
                if len(drop_idxs):
                    gdf.drop(drop_idxs, inplace=True)
                    dropped_idxs.extend(drop_idxs)

        # check if any fields remain
        if len(gdf):
            # compile filtered predictions in array format
            masks_I = maskedArr[..., gdf.index]
            masks_I = [self._get_largest_cc(x) for x in masks_I.transpose(2, 0, 1)]
            masks_I = np.stack(masks_I).transpose(1, 2, 0)
            # intersect segments & remove silver polygons smaller than area threshold
            masks_II = label((masks_I * (np.arange(masks_I.shape[-1]) + 1)).sum(axis=2))
            masks_II = remove_small_objects(masks_II, self.area_t)
            self.pred_pp = np.where(masks_II > 0, masks_II, -1)

        else:
            self.pred_pp = -np.ones(maskedArr.shape[:2])

    def bbox_pp(self):
        """
        Postprocessing based on bbox IoUs,
        assumes that bbox NMS has already been applied to preds
        (i.e. running inference via test.py using cfg_options
        with model.test_cfg.rcnn.nms.iou_threshold)
        """
        # get scores & predictions
        scores = self.pred["pred_instances"]["scores"].numpy()
        masks = self.pred["pred_instances"]["masks"]

        # decode single 256x256 map to bitmap
        if torch.is_tensor(masks):
            maskedArr = np.transpose(masks.numpy().astype(np.int8), (1, 2, 0))
        else:
            maskedArr = cocomask.decode(masks)

        # remove segments below a certain score
        bool_score_idx = scores > self.score_t
        maskedArr = maskedArr[..., bool_score_idx]
        scores = scores[bool_score_idx]

        if len(scores):
            # suppress smaller objects
            masks_I = [self._get_largest_cc(x) for x in maskedArr.transpose(2, 0, 1)]
            masks_I = np.stack(masks_I).transpose(1, 2, 0)

            # intersect segments & remove silver polygons smaller than area threshold
            masks_II = label((masks_I * (np.arange(masks_I.shape[-1]) + 1)).sum(axis=2))
            masks_II = remove_small_objects(masks_II, 25)
            self.pred_pp = np.where(masks_II > 0, masks_II, -1)

        else:
            self.pred_pp = -np.ones(maskedArr.shape[:2])

    def write_to_disk(
        self,
        data_dir="d:/thesis/data/ai4boundaries/filtered",
        save_dir="d:/thesis/results/instance_seg/postprocessing",
    ):
        img_path = self.pred["img_path"]
        tile_path = os.path.join(data_dir, os.path.split(img_path)[-1])
        tile_name = os.path.split(img_path)[-1].split("_S2_10m_256.nc")[0]
        tile_ds = xr.open_dataset(tile_path)
        pred_pp = xr.DataArray(
            self.pred_pp.astype(np.int32),
            coords={"y": tile_ds.y, "x": tile_ds.x},
            dims=["y", "x"],
        )
        os.makedirs(save_dir, exist_ok=True)
        dst = os.path.join(save_dir, f"{tile_name}.tif")
        pred_pp.rio.to_raster(dst, compress="LZW")

    @staticmethod
    def _imports_for_spawned_processes():
        global gpd, np, pd, os, xr, torch
        global bitmap_to_polygon, cocomask
        global Polygon, MultiPolygon
        global label, remove_small_objects, regionprops

        import geopandas as gpd
        import pandas as pd
        import numpy as np
        import torch
        import os
        import pycocotools.mask as cocomask
        import xarray as xr
        from mmdet.visualization.local_visualizer import bitmap_to_polygon
        from shapely.geometry import Polygon, MultiPolygon
        from skimage.morphology import label, remove_small_objects
        from skimage.measure import regionprops

    @staticmethod
    def _convert_to_poly(arr_2d):
        """
        Coverts bitmaps corresponding to the
        instance segmentation output format to shapely polygons
        """
        poly_arr = bitmap_to_polygon(arr_2d)[0]
        poly_arr = [x for x in poly_arr if len(x) > 2]
        polygon_objects = [Polygon(p) for p in poly_arr]
        # clean up geometries by removing self-intersections
        polygon_objects = [p.buffer(0) for p in polygon_objects]
        if len(polygon_objects):
            # simplify multipolygon geometries by taking largest prediction
            largest_idx = np.argmax([x.area for x in polygon_objects])
            return polygon_objects[largest_idx]
        else:
            # no meanigful polygons remaining after cleaning
            return Polygon()

    @staticmethod
    def _get_largest_cc(arr_2d):
        """
        Given a labelled 2d array it extracts the largest connected component
        per category, such that only one region remains for each integer
        """
        if arr_2d.sum():
            labelled = label(arr_2d, connectivity=1)
            rprops = regionprops(labelled)
            arr_filtered = labelled == (1 + np.argmax([i.area for i in rprops]))
            return arr_filtered.astype(int)
        else:
            return arr_2d.astype(int)

    @staticmethod
    def pp_single(
        pred,
        pp_type="mask",
        data_dir="d:/thesis/data/ai4boundaries/filtered",
        save_dir="d:/thesis/results/instance_seg/postprocessing",
        **kwargs,
    ):
        """
        Runs post-processing pipeline on single prediction instance
        """
        ipp = InstancePostProcessor(pred, **kwargs)
        if pp_type == "mask":
            ipp.mask_pp()
        if pp_type == "bbox":
            ipp.bbox_pp()
        ipp.write_to_disk(data_dir, save_dir)

    @staticmethod
    def pp_parallel(
        preds,
        area_thres,
        score_thres,
        iou_thres,
        pp_type="mask",
        data_dir="d:/thesis/data/ai4boundaries/filtered",
        save_dir="d:/thesis/results/instance_seg/postprocessing",
        n_cores=os.cpu_count() - 1,
        verbose=False,
    ):
        """
        Runs post-processing on multiple predictions in thread-parallelised manner
        """
        ip = partial(
            InstancePostProcessor.pp_single,
            area_thres=area_thres,
            score_thres=score_thres,
            iou_thres=iou_thres,
            pp_type=pp_type,
            data_dir=data_dir,
            save_dir=save_dir,
        )

        tqdm_desc = f"Postprocessing for {len(preds)} tiles"
        with multiprocess.Pool(processes=n_cores) as pool:
            pool.map(
                ip,
                tqdm(
                    preds,
                    total=len(preds),
                    disable=not verbose,
                    smoothing=0,
                    desc=tqdm_desc,
                ),
                chunksize=1,
            )
            pool.close()
            pool.join()
