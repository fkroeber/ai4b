import multiprocess
import numpy as np
import os
from functools import partial
from tqdm import tqdm


class CannyWater:
    """
    canny edge-detection based hierarchical watershed segmentation,
    sequence of canny followed by hierarchical watershed tree construction & horizontal cuts,
    supervised parametrisation of both parts based on provided training samples
    """

    # define optimisation params as class attribute
    opt_params = [
        "preprocess_method",
        "grad_method",
        "sigma",
        "hysteresis_thres",
        "segment_method",
        "h_thres",
        "area_thres",
        "compactness_weight",
        "epsilon",
    ]

    def __init__(self, tile_name, data_dir="d:/thesis/data/ai4boundaries/filtered"):
        self._imports_for_spawned_processes()
        self.tile_name = tile_name
        self.tile_path = os.path.join(data_dir, f"{tile_name}_S2_10m_256.nc")
        self.label_path = os.path.join(data_dir, f"{tile_name}_S2label_10m_256.tif")

    def read(self):
        """
        Read a single sample & its corresponding label
        """
        # read satellite imagery
        tile_ds = xr.open_dataset(self.tile_path)
        proj_wkt = tile_ds["spatial_ref"].attrs["spatial_ref"]
        tile_ds.rio.write_crs(proj_wkt, inplace=True)
        self.tile_ds = tile_ds

        # modify satellite imagery by rescaling & clipping optical bands to [0,1]
        # also replace nan in NDVI by 0
        tile_arr = np.array(self.tile_ds.to_array()).reshape(-1, 256, 256)
        tile_arr[:24, ...] = tile_arr[:24, ...] / 10000
        tile_arr[:24, ...] = np.where(tile_arr[:24, ...] > 1, 1, tile_arr[:24, ...])
        self.tile_arr = np.nan_to_num(tile_arr)

        # read labels
        label_ds = xr.open_dataset(self.label_path)
        proj_wkt = label_ds["spatial_ref"].attrs["spatial_ref"]
        label_ds.rio.write_crs(proj_wkt, inplace=True)

        # store extent mask & field enumeration as relevant label info
        # extent equals np.where(np.nan_to_num(cw.field_enum,0) > 0, 1, 0)
        self.extent_mask = np.array(label_ds["band_data"][0, :, :])
        self.field_enum = np.array(label_ds["band_data"][3, :, :])

    def preprocess(self, method=None):
        """
        Apply (optional) image enhancements techniques
        """
        # requires order of bands to be C x H x W
        # perform fuzzy histogram hyperbolisation
        if method == "FHH":
            self.tile_arr = np.stack([self._fhh(x) for x in self.tile_arr])

        # contrast-limited adaptive histogram equalisation
        elif method == "CLAHE":
            rescaled = [equalize_adapthist(x, kernel_size=64) for x in self.tile_arr]
            self.tile_arr = np.stack(rescaled)

        else:
            pass

    def calc_grad(self, method="post_avg_soft", **kwargs):
        """
        Retrieve a single band canny gradient layer from the timeseries stack of input bands,
        several options can be chosen to average the results from single input bands
        * post_avg_crisp: calculate canny edges on each map & average crisp labels
        * post_avg_soft: calculate gradient for canny edges on each map & average soft labels
        * pre_avg_soft: calculate gradient for each map, average, perform NMS & hysteresis thresholding subsequently
        * rgb_post_avg_soft: calculate gradient based on 3-channel inputs (true & false color composites), average soft labels
        """

        if method == "post_avg_crisp":
            edge_maps = list(map(partial(canny, **kwargs), self.tile_arr))
            grad_map = np.stack(edge_maps).mean(axis=0)

        elif method == "post_avg_soft":
            grad_maps = list(map(partial(canny_grad, **kwargs), self.tile_arr))
            grad_map = np.stack(grad_maps).mean(axis=0)

        elif method == "pre_avg_soft":
            grad_maps = list(map(partial(sobel_grad, **kwargs), self.tile_arr))
            grad_maps = np.stack(grad_maps)
            isobel = grad_maps[:, 0, :, :].mean(axis=0)
            jsobel = grad_maps[:, 1, :, :].mean(axis=0)
            magnitudes = grad_maps[:, 2, :, :].mean(axis=0)
            eroded_mask = grad_maps[0, 3, :, :].astype(np.bool_)
            grad_map = canny_postprocess(
                isobel, jsobel, magnitudes, eroded_mask, **kwargs
            )

        elif method == "rgb_post_avg_soft":
            rgb_idxs = np.arange(self.tile_arr.shape[0]).reshape(5, 6)[:3, :][::-1]
            rgb_idxs = [rgb_idxs[:, i] for i in range(rgb_idxs.shape[1])]
            rgbs = [255 * self.tile_arr[x] for x in rgb_idxs]
            rgbs = [np.transpose(rgb, (1, 2, 0)) for rgb in rgbs]
            nir_idxs = np.arange(self.tile_arr.shape[0]).reshape(5, 6)[1:4, :][::-1]
            nir_idxs = [nir_idxs[:, i] for i in range(nir_idxs.shape[1])]
            nirs = [255 * self.tile_arr[x] for x in nir_idxs]
            nirs = [np.transpose(nir, (1, 2, 0)) for nir in nirs]
            grad_maps = list(map(partial(detect_rgb, **kwargs), [*rgbs, *nirs]))
            grad_map = np.stack(grad_maps).mean(axis=0)

        self.grad_map = pad_edge(grad_map)

    def segment(self, method="dynamics_hcut", h_thres=0.1, area_thres=25, **kwargs):
        """
        Get a labelled prediction by performing a hierarchical watershed cut,
        different strategies for performing the cut:
        * dynamics_hcut: horizontal cut based on dynamics
        * combined_hcut: horizontal cut based on combined dynamics & compactness saliency
        * compact_energy_cut: optimal energy cut hierachy transformation followed by horizontal cut
            + data fidelity energy: mean NDVI variability
            + regularisation energy: compactness
        * ms_energy_cut: optimal energy cut hierachy transformation followed by horizontal cut
            + energies correspond to Mumford-Shah energy
            + data fidelity energy: mean NDVI variability
            + regularisation energy: contour length
        """
        # common basis - create weighted graph & watershed tree with 4-connectivity
        self.size = self.grad_map.shape[:2]
        self.graph = hg.get_4_adjacency_graph(self.size)
        edge_weights = hg.weight_graph(
            self.graph, self.grad_map, hg.WeightFunction.mean
        )
        tree, altitudes = hg.watershed_hierarchy_by_dynamics(self.graph, edge_weights)

        # filter small nodes from tree
        tree, altitudes = hg.filter_small_nodes_from_tree(tree, altitudes, area_thres)

        if method == "dynamics_hcut":
            # get segments by cutting the tree by its altitude
            # note that this is equivalent to the following but additionally gives the saliency map
            # labelisation = hg.labelisation_horizontal_cut_from_threshold(tree, altitudes, threshold=h_thres)
            saliency = hg.saliency(tree, altitudes)
            saliency[saliency < h_thres] = 0
            labelisation = hg.labelisation_watershed(self.graph, saliency)

        elif method == "combined_hcut":
            # 1st watershed saliency based on dynamics
            saliency = hg.saliency(tree, altitudes)
            grid_I = hg.graph_4_adjacency_2_khalimsky(self.graph, saliency)

            # calculate compactness
            # use topological height for weighting of shape-based saliencies
            heights = hg.attribute_topological_height(tree)
            heights_relative = heights / heights.max()
            compact = self._compactness(tree) * heights_relative

            # 2nd saliency based on compactness
            compact[heights == 0] = 0
            saliency = hg.accumulate_on_contours(tree, compact, hg.Accumulators.max)
            grid_II = hg.graph_4_adjacency_2_khalimsky(self.graph, saliency)

            # combination of saliency maps
            compact_weight = kwargs.get("compact_weight")
            smap = (grid_I + compact_weight * grid_II) / (1 + compact_weight)

            # get segments by cutting the tree using saliency
            _, saliency = hg.khalimsky_2_graph_4_adjacency(smap)
            saliency[saliency < h_thres] = 0
            labelisation = hg.labelisation_watershed(self.graph, saliency)

        elif method == "compact_energy_cut":
            # transform hierarchy into optimal energy cut hierarchy
            compact = self._compactness(tree)
            mean_ndvi = self.tile_arr[24:, ...].mean(axis=0)
            mean, covar = hg.attribute_gaussian_region_weights_model(tree, mean_ndvi)
            data_fidelity = covar * hg.attribute_area(tree)
            ms_tree, ms_alt = hg.hierarchy_to_optimal_energy_cut_hierarchy(
                tree, data_fidelity, compact
            )

            # create & threshold saliency map
            saliency = hg.accumulate_on_contours(ms_tree, ms_alt, hg.Accumulators.max)
            saliency[saliency < h_thres] = 0
            labelisation = hg.labelisation_watershed(self.graph, saliency)

        elif method == "ms_energy_cut":
            # transform hierarchy into optimal energy cut hierarchy
            mean_ndvi = self.tile_arr[24:, ...].mean(axis=0)
            ms_tree, ms_alt = hg.hierarchy_to_optimal_MumfordShah_energy_cut_hierarchy(
                tree, mean_ndvi
            )

            # create & threshold saliency map
            saliency = hg.accumulate_on_contours(ms_tree, ms_alt, hg.Accumulators.max)
            saliency[saliency < h_thres] = 0
            labelisation = hg.labelisation_watershed(self.graph, saliency)

        self.saliency = saliency
        self.segments = labelisation.astype(np.int64)

    def postprocess(self, epsilon=0):
        """
        Apply contour smooting (Douglas-Peucker simplification)
        """
        watershed_cut = hg.weight_graph(self.graph, self.segments, hg.WeightFunction.L0)
        contours = hg.fit_contour_2d(self.graph, self.size, watershed_cut)
        contours.subdivide(epsilon=epsilon, min_size=0, relative_epsilon=False)
        contours_simpl = ~self._draw_contour(contours, self.size)
        graph = hg.get_4_adjacency_graph(self.size)
        edge_weights = hg.weight_graph(graph, contours_simpl, hg.WeightFunction.mean)
        labelisation = hg.labelisation_watershed(graph, edge_weights)
        self.segments = labelisation.astype(np.int64)

    def evaluate(self, metrics, agg=True):
        """
        Calculate segmentation accuracy metrics
        """
        self.eval_m = Evaluator(self.segments, self.field_enum, metrics, agg)
        self.eval_m.eval_tile()

    @staticmethod
    def obj_fun(
        hyperparams, train_sample, data_dir="d:/thesis/data/ai4boundaries/filtered"
    ):
        """
        Function that runs the whole pipeline based on a set of specified parameters,
        return a loss score that can be used in a supervised setting to optimise the parameters
        """
        # get named hyperparameters
        params = {}
        for i, param in enumerate(CannyWater.opt_params):
            params[param] = hyperparams[i]
        # run pipeline with specified params
        cw = CannyWater(train_sample, data_dir)
        cw.read()
        cw.preprocess(method=params["preprocess_method"])
        cw.calc_grad(
            method=params["grad_method"],
            sigma=params["sigma"],
            low_threshold=0.3 * params["hysteresis_thres"],
            high_threshold=params["hysteresis_thres"],
        )
        cw.segment(
            method=params["segment_method"],
            h_thres=params["h_thres"],
            area_thres=params["area_thres"],
            compact_weight=params["compactness_weight"],
        )
        if params["epsilon"] > 0:
            cw.postprocess(params["epsilon"])
        # calculate loss
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cw.evaluate(["f1", "f1_w"])
            f1 = float(cw.eval_m.stats_agg["f1"])
            f1w = float(cw.eval_m.stats_agg["f1_w"])
            loss = -(f1 + f1w) / 2
            loss = loss if ~np.isnan(loss) else 0
        return loss

    @staticmethod
    def obj_fun_parallel(
        samples,
        hyperparams,
        data_dir="d:/thesis/data/ai4boundaries/filtered",
        n_cores=os.cpu_count() - 1,
        verbose=False,
    ):
        """
        Runs objective_fun method on all training samples using multiple threads
        """
        cw = partial(CannyWater.obj_fun, data_dir=data_dir)
        with multiprocess.Pool(processes=n_cores) as pool:
            losses = pool.starmap(
                lambda fun, params, tile: fun(params, tile),
                tqdm(
                    zip(
                        len(samples) * [cw],
                        len(samples) * [hyperparams],
                        samples,
                    ),
                    total=len(samples),
                    disable=not verbose,
                    smoothing=0,
                ),
                chunksize=1,
            )
            pool.close()
            pool.join()
        return np.mean(losses)

    @staticmethod
    def assess_sample(
        hyperparams,
        metrics,
        sample,
        agg=False,
        data_dir="d:/thesis/data/ai4boundaries/filtered",
        only_metrics=True,
    ):
        """
        Function that runs the whole pipeline based on a set of specified parameters,
        returns either the whole cws instance or only the evaluation metrics sub-element
        """
        # get named hyperparameters
        params = {}
        for i, param in enumerate(CannyWater.opt_params):
            params[param] = hyperparams[i]
        # run pipeline with specified params
        cw = CannyWater(sample, data_dir)
        cw.read()
        cw.preprocess(method=params["preprocess_method"])
        cw.calc_grad(
            method=params["grad_method"],
            sigma=params["sigma"],
            low_threshold=0.3 * params["hysteresis_thres"],
            high_threshold=params["hysteresis_thres"],
        )
        cw.segment(
            method=params["segment_method"],
            h_thres=params["h_thres"],
            area_thres=params["area_thres"],
            compact_weight=params["compactness_weight"],
        )
        if params["epsilon"] > 0:
            cw.postprocess(params["epsilon"])
        # calculate metrics
        cw.evaluate(metrics, agg=agg)
        # return result
        if only_metrics:
            return [sample, cw.eval_m]
        else:
            return cw

    @staticmethod
    def assess_parallel(
        samples,
        hyperparams,
        metrics,
        agg=False,
        data_dir="d:/thesis/data/ai4boundaries/filtered",
        n_cores=os.cpu_count() - 1,
        verbose=False,
    ):
        """
        Runs assessment method using multiple threads
        """
        cw = partial(
            CannyWater.assess_sample, data_dir=data_dir, agg=agg, only_metrics=True
        )
        tqdm_desc = f"Calc segmentation metrics for {len(samples)} tiles"
        with multiprocess.Pool(processes=n_cores) as pool:
            eval_metrics = pool.starmap(
                lambda fun, params, metrics, tile: fun(params, metrics, tile),
                tqdm(
                    zip(
                        len(samples) * [cw],
                        len(samples) * [hyperparams],
                        len(samples) * [metrics],
                        samples,
                    ),
                    total=len(samples),
                    disable=not verbose,
                    smoothing=0,
                    desc=tqdm_desc,
                ),
                chunksize=1,
            )
            pool.close()
            pool.join()
        return eval_metrics

    def _compactness(self, tree):
        return 4 * np.pi * hg.attribute_compactness(tree, normalize=False)

    def _draw_contour(self, contour, size):
        image = np.zeros(size, np.uint8) + 255
        image_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(image_pil)
        for polyline in contour:
            for segment in polyline:
                p1 = segment[0][1]
                p2 = segment[len(segment) - 1][1]
                draw.line((p1[1], p1[0], p2[1], p2[0]), fill=0)
        del draw
        return np.array(image_pil)

    def _fhh(self, img):
        minVal, maxVal = img.min(), img.max()
        greyLevel = (img - minVal) / (maxVal - minVal)
        greyLevel = np.power(greyLevel, 0.8)
        rescaled_img = (1 / (np.exp(-1) - 1)) * (np.exp(-greyLevel) - 1)
        return rescaled_img

    def _imports_for_spawned_processes(self):
        global cv2, hg, np, os, sys, rs, rxr, warnings, xr
        global Evaluator, equalize_adapthist, normalise, pad_edge, tqdm
        global canny, canny_grad, canny_postprocess, detect_rgb, sobel_grad
        global Image, ImageDraw

        import cv2
        import higra as hg
        import numpy as np
        import os
        import sys
        import rasterio as rs
        import rioxarray as rxr
        import warnings
        import xarray as xr

        from eval.segment import Evaluator
        from skimage.exposure import equalize_adapthist
        from skimage.feature import canny
        from PIL import Image, ImageDraw
        from tqdm import tqdm
        from gradient import canny_grad, sobel_grad, canny_postprocess, detect_rgb
        from utils.common import normalise, pad_edge
