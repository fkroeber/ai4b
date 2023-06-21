import geopandas as gpd
import numpy as np
import torch
import torchvision
import pandas as pd
import shapely
import warnings
from utils.geo import rs_to_gdf

from skimage.morphology import dilation, square
from skimage.segmentation import find_boundaries
from sklearn.metrics import (
    adjusted_rand_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.metrics.cluster import v_measure_score
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# list available metrics (either segmentation or edge-specific)
# raster-based
m_edges = ["f1", "precision", "recall", "f1_w", "f1_b", "rand", "v_measure"]
m_image = ["lpips", "ssmi"]
m_raster = [*m_edges, *m_image]
# object-based
m_segments_mask = ["iou", "io_ref", "io_pred", "iou_boundary"]
m_segments_edge = ["mae_gt", "mae_pred", "bde", "hausdorff_dist"]
m_segments_centroid = ["pos_sim", "centroid_dist"]
m_objects = [*m_segments_mask, *m_segments_edge, *m_segments_centroid]


class Evaluator:
    """
    Evaluate the boundary prediction (preds) for a given tile by comparing it to
    the ground truth (gt) using a range of available metrics
    """

    def __init__(self, pred, gt, metrics, agg=True):
        """
        pred: predictions as categorial raster HxW
        gt: ground truth as categorial raster HxW
        metrics: list of metrics to be produced
        agg: aggregation per tile for object-based stats
            + if True: creates a single dict with metrics aggregated for all fields
            + if False: leaves df unaggregated with metrics for each field individually
        """
        self.pred = pred
        self.gt = gt
        self.metrics = metrics
        self.agg = agg

    def eval_tile(self):
        """
        Runs the main script, i.e. calculates various metrics
        for evaluation of the segmentation result
        """

        # calculate different similarity metrics
        if any(i in self.metrics for i in m_objects):
            self._object_based_stats()

        if any(i in self.metrics for i in m_raster):
            self._raster_based_stats()

        if self.agg:
            self.aggregate()

    def aggregate(self):
        """
        Aggregates stats tile-wise,
        only applies to object stats,
        raster stats already aggregated in first place
        """
        self.stats_agg = {}
        if any(i in self.metrics for i in m_objects):
            obj_stats_avg = self.stats_objects.mean(numeric_only=True)
            for m in self.metrics:
                if m in obj_stats_avg:
                    self.stats_agg[m] = obj_stats_avg[m]

        if any(i in self.metrics for i in m_raster):
            self.stats_agg.update(self.stats_raster)

    def _raster_based_stats(self):
        """
        Fast to calculate, always returned in aggregated format per tile
        """

        # preprocessing steps
        if any(i in self.metrics for i in m_raster):
            # binarize prediction & ground truth into field vs. boundaries
            pred_binary = find_boundaries(self.pred)
            gt_binary = find_boundaries(np.nan_to_num(self.gt))
            # restrict eval to agricultural mask (fields & boundaries vs. non-fields)
            # dilate to include exterior boundaries when masking subsequently
            gt_mask = dilation(np.nan_to_num(self.gt) > 0, square(3))

        if any(i in self.metrics for i in m_image):
            # transform binary masks into tensors
            img_pred = torch.tensor(np.where(gt_mask, pred_binary, 0))
            img_pred = img_pred.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
            img_gt = torch.tensor(np.where(gt_mask, gt_binary, 0))
            img_gt = img_gt.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)

        # mind right order of pred - target since this varies across sklearn & torchmetrics
        self.stats_raster = {}
        if "f1" in self.metrics:
            # boundary F1
            self.stats_raster["f1"] = f1_score(gt_binary[gt_mask], pred_binary[gt_mask])

        if "f1_b" in self.metrics:
            # inspired by Martin et al. 2004 - boundary F1 with distance tolerance
            # compute dilations of preds & gts to tolerate small boundary errors
            pred_dilated = dilation(pred_binary, square(2))
            gt_dilated = dilation(gt_binary, square(2))
            # compute precision & recall
            p = precision_score(gt_dilated[gt_mask], pred_binary[gt_mask])
            r = recall_score(gt_binary[gt_mask], pred_dilated[gt_mask])
            # equivalent to..
            # p = sum((pred_binary & gt_dilated)[gt_mask])/sum(pred_binary[gt_mask])
            # r = sum((gt_binary & pred_dilated)[gt_mask])/sum(gt_binary[gt_mask])
            self.stats_raster["f1_b"] = 2 * p * r / (p + r) if p + r > 0 else np.nan

        if "f1_w" in self.metrics:
            # overall F1 for both classes balanced by their frequency
            # similar to HED loss (auto-balance class inequality)
            n_neg, n_pos = np.unique(gt_binary[gt_mask], return_counts=True)[1]
            w_pos = n_neg / (n_neg + n_pos)
            w_neg = n_pos / (n_neg + n_pos)
            w_array = np.where(gt_binary[gt_mask], w_pos, w_neg)
            self.stats_raster["f1_w"] = f1_score(
                gt_binary[gt_mask],
                pred_binary[gt_mask],
                average="weighted",
                sample_weight=w_array,
            )

        if "precision" in self.metrics:
            self.stats_raster["precision"] = precision_score(
                gt_binary[gt_mask], pred_binary[gt_mask]
            )

        if "recall" in self.metrics:
            self.stats_raster["recall"] = recall_score(
                gt_binary[gt_mask], pred_binary[gt_mask]
            )

        if "rand" in self.metrics:
            self.stats_raster["rand"] = adjusted_rand_score(
                self.gt[gt_mask].astype(np.int32), self.pred[gt_mask].astype(np.int32)
            )

        if "v_measure" in self.metrics:
            self.stats_raster["v_measure"] = v_measure_score(
                self.gt[gt_mask].astype(np.int32), self.pred[gt_mask].astype(np.int32)
            )

        if "lpips" in self.metrics:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lpips = LearnedPerceptualImagePatchSimilarity(
                    net_type="squeeze",
                    weights=torchvision.models.SqueezeNet1_1_Weights.DEFAULT,
                )
                self.stats_raster["lpips"] = lpips(img_pred, img_gt).item()

        if "ssmi" in self.metrics:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ssim = StructuralSimilarityIndexMeasure()
                self.stats_raster["ssmi"] = ssim(
                    img_pred.to(torch.float32), img_gt.to(torch.float32)
                ).item()

    def _object_based_stats(self):
        """
        Calculate stats for masks with highest IoU overlap relative to the reference.
        This way each reference object will only be evaluated once taking the best candidate among the predictions.
        Computationally less efficient than pixel-wise stats
        """
        # convert
        gr_df = rs_to_gdf(self.gt)
        pred_df = rs_to_gdf(self.pred)

        # calculate IoU for all overlapping segments
        # performance remark:
        # sjoin with underlying tree indexing & subsequent iteration over matches fastest
        # gdf.overlay(how="union"/"intersection") slower
        df_combi = gpd.sjoin(gr_df, pred_df, how="left", predicate="intersects")
        col_names = {"geometry": "gt_geom"}
        df_combi = df_combi.rename(columns=col_names)
        df_combi = pd.merge(df_combi, pred_df, left_on="index_right", right_index=True)
        col_names = {
            "geometry": "pred_geom",
            "shp_idx_left": "gt_idx",
            "shp_idx_right": "pred_idx",
        }
        df_combi = df_combi.rename(columns=col_names)
        df_combi = df_combi[["gt_idx", "gt_geom", "pred_idx", "pred_geom"]]
        intersection = df_combi["gt_geom"].intersection(df_combi["pred_geom"])
        union = df_combi["gt_geom"].union(df_combi["pred_geom"])

        # append field sizes
        df_combi["gt_area"] = df_combi["gt_geom"].area
        df_combi["pred_area"] = df_combi["pred_geom"].area

        # append iou metrics and over-/undersegmentation metrics
        df_combi["iou"] = intersection.area / union.area
        df_combi["io_ref"] = intersection.area / df_combi["gt_geom"].area
        df_combi["io_pred"] = intersection.area / df_combi["pred_geom"].area

        # filter delineated fields with maximum overlap for each validation field
        # amounts to non-maxima suppression in object detection
        idx_max_overlap = (
            df_combi.groupby(["gt_idx"])["iou"].transform(max) == df_combi["iou"]
        )
        df_combi = df_combi[idx_max_overlap]
        df_combi.reset_index(drop=True, inplace=True)

        # calculate boudary IoU
        if "iou_boundary" in self.metrics:
            dist = 2.0  # 20m edge tolerance
            gt, pred = df_combi["gt_geom"], df_combi["pred_geom"]
            gt_boundary = gt.difference(gt.buffer(-dist), align=False)
            pred_boundary = pred.difference(pred.buffer(-dist), align=False)
            intersection = gt_boundary.intersection(pred_boundary, align=False)
            union = gt_boundary.union(pred_boundary, align=False)
            df_combi["iou_boundary"] = intersection.area / union.area

        # calculate edge-based metrics for all fields
        if any(i in self.metrics for i in m_segments_edge):
            gt_ps = [self._gen_boundary_points(x) for x in df_combi["gt_geom"]]
            pred_ps = [self._gen_boundary_points(x) for x in df_combi["pred_geom"]]
            mae_gt_dist = [self._mae_edge(x, y) for x, y in zip(pred_ps, gt_ps)]
            mae_pred_dist = [self._mae_edge(x, y) for x, y in zip(gt_ps, pred_ps)]
            zipped_geoms = zip(df_combi["pred_geom"], df_combi["gt_geom"])
            hausdorff_dist = [x.hausdorff_distance(y) for x, y in zipped_geoms]
            df_combi["mae_gt"] = mae_gt_dist
            df_combi["mae_pred"] = mae_pred_dist
            df_combi["bde"] = (df_combi["mae_gt"] + df_combi["mae_pred"]) / 2
            df_combi["hausdorff_dist"] = hausdorff_dist

        # calculate centroid-based metrics
        if any(i in self.metrics for i in m_segments_centroid):
            c_dist = df_combi.gt_geom.centroid.distance(df_combi.pred_geom.centroid)
            df_combi["centroid_dist"] = c_dist

            if "pos_sim" in self.metrics:
                # Lizarazo 2014 - combined area circle based
                cac_area = df_combi.gt_geom.area + df_combi.pred_geom.area
                cac_diameter = 2 * np.sqrt(cac_area / np.pi)
                df_combi["pos_sim"] = 1 - c_dist / cac_diameter

        self.stats_objects = df_combi

    def _gen_boundary_points(self, geom, dist=2.5):
        """
        Creates points with a specified spacing along the boundary of a segment,
        Default spacing is 2.5m which allows sufficient approx. of actual result,
        Preprocessing for subsequent edge-based metrics
        """
        if type(geom) == shapely.geometry.polygon.Polygon:
            distances = np.arange(0, geom.length, dist)
            points = [geom.exterior.interpolate(distance) for distance in distances]
            multipoint = shapely.ops.unary_union(points)
        elif type(geom) == shapely.geometry.multipolygon.MultiPolygon:
            multiparts = [x.exterior for x in geom.geoms]
            distances = [np.arange(0, geom.length, dist) for geom in multiparts]
            ls = []
            for i in range(len(multiparts)):
                ls.append(
                    [multiparts[i].interpolate(distance) for distance in distances[i]]
                )
            multipoint = shapely.ops.unary_union(
                [item for sublist in ls for item in sublist]
            )
        return multipoint

    def _mae_edge(self, source_geom, replica_geom):
        """
        Calculates mean edge error by matching boundary points based on the shortest distance
        """
        dists = []
        for ref_point in source_geom.geoms:
            near_points = shapely.ops.nearest_points(ref_point, replica_geom)
            dists.append(near_points[0].distance(near_points[1]))
        return np.mean(np.array(dists))
