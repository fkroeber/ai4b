import networkx as nx
import os
import numpy as np
import pandas as pd
from skimage import util
import time
from functools import partial
from itertools import product
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import BaseCrossValidator, GridSearchCV, GroupKFold
from skimage.measure import regionprops_table
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import make_scorer
from obia.cws import CannyWater
from skimage.future import graph
import warnings
import multiprocess
from tqdm import tqdm


class FeatEngineer:
    """
    Feature engineering for tiles,
    object- or pixel-wise features are provided
    """

    def __init__(
        self,
        tile_name,
        cws_params,
        feat_set,
        env_covars=True,
        data_dir="d:/thesis/data/ai4boundaries/filtered",
        stats_csv="d:/thesis/results/eda/filtered_stats_tiles.csv",
    ):
        # copy args
        self.tile_name = tile_name
        self.cws_params = cws_params
        self.feat_set = feat_set
        self.env_covars = env_covars
        self.data_dir = data_dir
        self.stats_csv = stats_csv
        # silence warnings
        warnings.simplefilter(action="ignore", category=FutureWarning)

    def create_segments(self, metrics=[]):
        """
        Canny-watershed segmentation (-> cws.py),
        default without evaluation since segmentation only interim
        """
        self.cws = CannyWater.assess_sample(
            hyperparams=self.cws_params,
            sample=self.tile_name,
            data_dir=self.data_dir,
            metrics=metrics,
            only_metrics=False,
        )

    def obj_feat_eng(self, timing=False):
        """
        Calculates a set of at most 65 object-specific features,
        Features to be calculated can be chosen and include...
            * 20 spectral temporal features
            * 8 shape features
            * 10 textural features
            * 27 contextural neighbourhood features
        Combines them with the ground truth label
        """
        if timing:
            _ = time.time()

        # check validity of called feature set
        valid_feat_sets = [
            ["spec"],
            ["spec", "geom"],
            ["spec", "geom", "text"],
            ["spec", "geom", "text", "neighbor"],
        ]

        if self.feat_set in valid_feat_sets:
            feat_set_idx = valid_feat_sets.index(self.feat_set)
        else:
            raise ValueError("Invalid feature set!")

        # I. get ground truth label
        def gt_label(regionmask, extent_mask):
            gt_field_prop = np.sum(extent_mask) / np.sum(regionmask)
            return gt_field_prop > 0.5

        labels = regionprops_table(
            self.cws.segments.astype(np.int64),
            intensity_image=self.cws.extent_mask,
            properties=["label"],
            extra_properties=(gt_label,),
        )
        labels = pd.DataFrame(labels).set_index("label")

        if timing:
            print(f"gt: {time.time() - _}")

        # II. spectral features (Waldner 2015)
        if "spec" in self.feat_set:
            # get bands decisive for temporal feature creation
            green_bands = self.cws.tile_arr[6:12]
            red_bands = self.cws.tile_arr[12:18]
            nir_bands = self.cws.tile_arr[18:24]
            ndvi_bands = self.cws.tile_arr[24:]
            ndvi_diff = (ndvi_bands - np.roll(ndvi_bands, 2, axis=0))[2:, ...]

            # create grid with indices
            m, n = ndvi_bands.shape[1:]
            I, J = np.ogrid[:m, :n]

            # get temporal steps of interest
            max_red = np.argmax(red_bands, axis=0)
            min_ndvi = np.argmin(ndvi_bands, axis=0)
            max_ndvi = np.argmax(ndvi_bands, axis=0)
            max_ndvi_incr = np.argmax(ndvi_diff, axis=0) + 1
            max_ndvi_decr = np.argmin(ndvi_diff, axis=0) + 1

            # get corresponding values of interest
            # bands: green, red, near infrared & NDVI
            # note: SWIR not available, thus replaced by NDVI
            bands = ["green_bands", "red_bands", "nir_bands", "ndvi_bands"]
            temp_steps = [
                "max_red",
                "min_ndvi",
                "max_ndvi",
                "max_ndvi_incr",
                "max_ndvi_decr",
            ]
            spatio_temp_feats = np.zeros((20, m, n))

            for i, (band, temp) in enumerate(product(bands, temp_steps)):
                spatio_temp_feats[i] = eval(f"{band}[{temp},I,J]")

            spec_feats = regionprops_table(
                self.cws.segments.astype(np.int64),
                intensity_image=np.transpose(spatio_temp_feats, (1, 2, 0)),
                properties=["label", "intensity_mean"],
            )
            spec_feats = pd.DataFrame(spec_feats).set_index("label")
            col_names = [
                f"{b.split('_')[0]}_{t}" for b, t in list(product(bands, temp_steps))
            ]
            spec_feats.columns = col_names

            if timing:
                print(f"spectral: {time.time() - _}")

        # III. shape features
        if "geom" in self.feat_set:
            geom_feats = regionprops_table(
                self.cws.segments.astype(np.int64),
                properties=[
                    "label",
                    "area",
                    "moments_hu",
                ],
            )
            geom_feats = pd.DataFrame(geom_feats).set_index("label")

            if timing:
                print(f"shape: {time.time() - _}")

        # IV. textural features
        if "text" in self.feat_set:
            # calculate texture measures only for NDVI bands
            ndvis = spatio_temp_feats[15:]

            text_feats = regionprops_table(
                self.cws.segments.astype(np.int64),
                intensity_image=np.transpose(ndvis, (1, 2, 0)),
                properties=[
                    "label",
                ],
                extra_properties=(self._entropy, self._stdv),
            )
            text_feats = pd.DataFrame(text_feats).set_index("label")
            col_names = [
                f"{x.split('-')[0][1:]}_{temp_steps[int(x.split('-')[1])]}"
                for x in text_feats.columns
            ]
            text_feats.columns = col_names

            if timing:
                print(f"textural: {time.time() - _}")

        # V. contextual features
        if "neighbor" in self.feat_set:
            # get properties of neighbours that are of interest
            ndvi_cols = [x for x in spec_feats.columns if x.split("_")[0] == "ndvi"]
            entropy_cols = [
                x for x in text_feats.columns if x.split("_")[0] == "entropy"
            ]

            neighbor_attrs = pd.concat(
                [
                    spec_feats[ndvi_cols],
                    geom_feats.iloc[:, :3],
                    text_feats[entropy_cols],
                ],
                axis=1,
            )

            # create region adjacency graph & matrix
            rag = graph.RAG(self.cws.segments)
            adj_matrix = nx.adjacency_matrix(rag)
            num_neighbors = adj_matrix.sum(axis=1).A1

            # convert neighbor_attrs to a numpy array
            neighbor_attrs_array = neighbor_attrs.reindex(list(rag.nodes)).to_numpy()

            # get mean vals
            neighbor_sum = adj_matrix.dot(neighbor_attrs_array)
            neighbor_mean = neighbor_sum / (num_neighbors.reshape(-1, 1))

            # get std vals
            neighbor_attrs_sq_sum = adj_matrix.dot(np.square(neighbor_attrs_array))
            num_neighbors_bessel = np.maximum(num_neighbors - 1, 1)[:, None]
            neighbor_attrs_sq_sum_bessel = np.divide(
                neighbor_attrs_sq_sum,
                num_neighbors_bessel,
                where=num_neighbors_bessel != 0,
            )
            neighbor_var = neighbor_attrs_sq_sum_bessel - np.divide(
                np.square(neighbor_sum),
                num_neighbors[:, None] * num_neighbors_bessel,
                where=num_neighbors_bessel != 0,
            )
            neighbor_std = np.sqrt(neighbor_var)

            # combine the results
            neighbor_data = np.column_stack(
                [rag.nodes, num_neighbors, neighbor_mean, neighbor_std]
            )
            neighbor_attrs_df = pd.DataFrame(neighbor_data).set_index(0)
            neighbor_feats = neighbor_attrs_df.fillna(0)
            neighbor_feats.index.names = ["label"]
            neighbor_feats.columns = [
                f"neighbours_{i}" for i in np.arange(len(neighbor_feats.columns))
            ]

            if timing:
                print(f"contextual: {time.time() - _}")

        # merge all information
        if feat_set_idx == 0:
            all_feats = [spec_feats]
        elif feat_set_idx == 1:
            all_feats = [spec_feats, geom_feats]
        elif feat_set_idx == 2:
            all_feats = [spec_feats, geom_feats, text_feats]
        elif feat_set_idx == 3:
            all_feats = [spec_feats, geom_feats, text_feats, neighbor_feats]

        feats_df = pd.concat([labels, *all_feats], axis=1).reset_index()

        # add area as informative attribute for evaluation
        if "area" not in feats_df.columns:
            area_feats = regionprops_table(
                self.cws.segments.astype(np.int64),
                properties=[
                    "label",
                    "area",
                ],
            )
            area_feats = pd.DataFrame(area_feats)["area"]
            feats_df = pd.concat([feats_df, area_feats], axis=1).reset_index(drop=True)

        # create uid
        feats_df.rename(columns={"label": "segment_id"}, inplace=True)
        feats_df.insert(0, "tile", self.tile_name)
        zipped = zip(feats_df.tile, feats_df.segment_id)
        uids = [f"{x}_{int(y)}" for x, y in zipped]
        feats_df.insert(0, "uid", uids)
        self.feats_df = feats_df.set_index("uid")

        if self.env_covars:
            self._add_env_covars()

        if timing:
            print(f"total: {time.time() - _}")

    def pxls_feat_eng(self, timing=False):
        """
        Calculates a set of at most 30 pixel-wise features,
        Features to be calculated can be chosen and include...
            * 20 spectral temporal features
            * 10 textural features
        Combines them with the ground truth label
        """
        if timing:
            _ = time.time()

        # check validity of called feature set
        valid_feat_sets = [
            ["spec"],
            ["spec", "text"],
        ]

        if self.feat_set in valid_feat_sets:
            pass
        else:
            raise ValueError("Invalid feature set!")

        # get bands decisive for temporal feature creation
        green_bands = self.cws.tile_arr[6:12]
        red_bands = self.cws.tile_arr[12:18]
        nir_bands = self.cws.tile_arr[18:24]
        ndvi_bands = self.cws.tile_arr[24:]
        ndvi_diff = (ndvi_bands - np.roll(ndvi_bands, 2, axis=0))[2:, ...]

        # create grid with indices
        m, n = ndvi_bands.shape[1:]
        I, J = np.ogrid[:m, :n]

        # get temporal steps of interest
        max_red = np.argmax(red_bands, axis=0)
        min_ndvi = np.argmin(ndvi_bands, axis=0)
        max_ndvi = np.argmax(ndvi_bands, axis=0)
        max_ndvi_incr = np.argmax(ndvi_diff, axis=0) + 1
        max_ndvi_decr = np.argmin(ndvi_diff, axis=0) + 1

        # get corresponding values of interest
        # bands: green, red, near infrared & NDVI
        # note: SWIR not available, thus replaced by NDVI
        bands = ["green_bands", "red_bands", "nir_bands", "ndvi_bands"]
        temp_steps = [
            "max_red",
            "min_ndvi",
            "max_ndvi",
            "max_ndvi_incr",
            "max_ndvi_decr",
        ]
        spatio_temp_feats = np.zeros((20, m, n))

        if "spec" in self.feat_set:
            for i, (band, temp) in enumerate(product(bands, temp_steps)):
                spatio_temp_feats[i] = eval(f"{band}[{temp},I,J]")

            st_cols = [
                f"{b.split('_')[0]}_{t}" for b, t in list(product(bands, temp_steps))
            ]

            feats_cube = spatio_temp_feats
            feats_df = pd.DataFrame(feats_cube.reshape(20, -1).T)
            feats_df.columns = st_cols

            if timing:
                print(f"spectral: {time.time() - _}")

        if "text" in self.feat_set:
            # calculate texture measures only for NDVI bands
            ndvis = spatio_temp_feats[15:]
            texture_s = np.stack([self._moving_std(ndvi, 3) for ndvi in ndvis])
            texture_m = np.stack([self._moving_std(ndvi, 5) for ndvi in ndvis])
            text_cols = [
                *[f"text_s{i}" for i in np.arange(ndvis.shape[0])],
                *[f"text_m{i}" for i in np.arange(ndvis.shape[0])],
            ]

            feats_cube = np.vstack([spatio_temp_feats, texture_s, texture_m])
            feats_df = pd.DataFrame(feats_cube.reshape(30, -1).T)
            feats_df.columns = [*st_cols, *text_cols]

            if timing:
                print(f"textural: {time.time() - _}")

        feats_df.insert(0, "gt_label", self.cws.extent_mask.flatten())
        feats_df.insert(0, "segment_id", self.cws.segments.flatten())
        feats_df.insert(0, "tile", self.tile_name)
        zipped = zip(feats_df.tile, feats_df.index.values)
        uids = [f"{x}_{int(y)}" for x, y in zipped]
        feats_df.insert(0, "uid", uids)
        self.feats_df = feats_df.set_index("uid")

        if self.env_covars:
            self._add_env_covars()

        if timing:
            print(f"total: {time.time() - _}")

    def _add_env_covars(self):
        # get pre-calculated covariats
        stats_tiles = pd.read_csv(self.stats_csv, index_col=0, header=[0, 1])
        stats_tiles.columns = stats_tiles.columns.get_level_values(1)
        biogeo_dummy = pd.get_dummies(stats_tiles["biogeo_region"], drop_first=False)
        self.env_covs = stats_tiles[["prec", "temp"]].join(biogeo_dummy)
        # add them to existing feature set
        self.feats_df = pd.merge(
            self.feats_df,
            self.env_covs,
            left_on="tile",
            right_index=True,
        )

    @staticmethod
    def eng_single(feat_type="obj_feats", only_feats=True, cws_metrics=[], **kwargs):
        """
        Wrapper to execute feature engineering pipeline,
        returns either only features or whole feat engineering object
        """
        rf_eng = FeatEngineer(**kwargs)
        rf_eng.create_segments(metrics=cws_metrics)
        if feat_type == "obj_feats":
            rf_eng.obj_feat_eng()
        if feat_type == "pxls_feats":
            rf_eng.pxls_feat_eng()
        if only_feats:
            return rf_eng.feats_df
        else:
            return rf_eng

    @staticmethod
    def eng_parallel(
        samples,
        cws_params,
        feat_set,
        feat_type="obj_feats",
        env_covars=True,
        data_dir="d:/thesis/data/ai4boundaries/filtered",
        stats_csv="d:/thesis/results/eda/filtered_stats_tiles.csv",
        only_feats=True,
        cws_metrics=[],
        n_cores=os.cpu_count() - 1,
        verbose=True,
    ):
        """
        Thread-parallelised execution of feature engineering,
        return compiled feature sets for all input tiles,
        note that the size of the returned feats may be quite large memory-wise
        (at most ~20 MB for pixelwise features for a single 256x256 map),
        risk of memory buffer error for large amounts of data
        """
        rf_eng = partial(
            FeatEngineer.eng_single,
            feat_type=feat_type,
            cws_metrics=cws_metrics,
            only_feats=only_feats,
            cws_params=cws_params,
            feat_set=feat_set,
            env_covars=env_covars,
            data_dir=data_dir,
            stats_csv=stats_csv,
        )

        def rf_wrapper(arg):
            return rf_eng(tile_name=arg)

        tqdm_desc = f"Feature engineering for {len(samples)} tiles"

        if only_feats:
            with multiprocess.Pool(processes=n_cores) as pool:
                feats = pool.map(
                    rf_wrapper,
                    tqdm(
                        samples,
                        total=len(samples),
                        disable=not verbose,
                        smoothing=0,
                        desc=tqdm_desc,
                    ),
                    chunksize=1,
                )
                pool.close()
                pool.join()
            return pd.concat(feats)

        else:
            with multiprocess.Pool(processes=n_cores) as pool:
                feat_engs = pool.map(
                    rf_wrapper,
                    tqdm(
                        samples,
                        total=len(samples),
                        disable=not verbose,
                        smoothing=0,
                        desc=tqdm_desc,
                    ),
                    chunksize=1,
                )
                pool.close()
                pool.join()
            return feat_engs

    @staticmethod
    def _entropy(regionmask, intensity_img):
        vals = intensity_img[regionmask]
        arr = stats.relfreq(vals, 100, defaultreallimits=(-1, 1))[0]
        return stats.entropy(arr)

    @staticmethod
    def _stdv(regionmask, intensity_img):
        vals = intensity_img[regionmask]
        std = np.std(vals)
        return std

    @staticmethod
    def _moving_std(image, window_size):
        pad_size = window_size // 2
        image = np.pad(image, pad_width=pad_size, mode="reflect")
        window_shape = (window_size, window_size)
        windowed_view = util.view_as_windows(image, window_shape)
        std_image = np.std(windowed_view, axis=(2, 3))
        return std_image


class RepeatedGroupKFold(BaseCrossValidator):
    """
    Repeated K-Fold carried out in groupwise seperate manner,
    suitable for spatial training data to avoid mixing of training & validation samples
    belonging to the same area (i.e. tile) -> obtain reliable estimate on generalisation capability
    """

    def __init__(self, n_splits=5, n_repeats=10, random_state=None):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits * self.n_repeats

    def split(self, X, y, groups):
        if groups is None:
            raise ValueError("Groups must be provided when using RepeatedGroupKFold")

        for i, repeat_idx in enumerate(range(self.n_repeats)):
            np.random.seed(i)
            shfl_idxs = np.random.permutation(len(X))
            X_shfl = X.iloc[shfl_idxs, :]
            y_shfl = y[shfl_idxs]
            groups_shfl = groups[shfl_idxs]
            group_kfold = GroupKFold(n_splits=self.n_splits)
            for train_idx, test_idx in group_kfold.split(X_shfl, y_shfl, groups_shfl):
                yield shfl_idxs[train_idx], shfl_idxs[test_idx]


class RFClassifier:
    """
    Random forest based binary classifier for segments,
    provides the following functionality:
        + performing hyperparameter search (pre-/post-aggregation per segment)
        + train rf (only for post-aggregation per segment)
        + apply & evaluate rf (only for post-aggregation per segment)
    """

    def __init__(self, X, y, tile_group):
        self.X = X
        self.y = y
        self.groups = tile_group

    def obj_hypertune(
        self,
        segment_area,
        gs_max_samples=2500,
        rgk_splits=5,
        rgk_repeats=10,
        n_jobs=-1,
        seed=42,
        verbose=False,
    ):
        """
        Hypertunes rf based on pre-aggregated segment values
        """
        self.weights = segment_area
        # set up rf
        rf = RandomForestClassifier(
            n_estimators=100, random_state=seed, n_jobs=n_jobs, class_weight="balanced"
        )
        # get subset of size gs_max_samples for grid search
        if len(self.y) > gs_max_samples:
            np.random.seed(seed)
            self.idx_subsets = np.random.choice(
                np.arange(len(self.y)), gs_max_samples, replace=False
            )
            self.X = self.X.iloc[self.idx_subsets, :]
            self.y = self.y[self.idx_subsets]
            self.groups = self.groups[self.idx_subsets]
        else:
            self.X = self.X
            self.y = self.y
            self.groups = self.groups
        # create multiple scoring metrics
        score_metrics = {
            "obj_acc": make_scorer(self.obj_acc_I, metric="acc"),
            "obj_f1": make_scorer(self.obj_acc_I, metric="f1"),
        }
        # set up hyperparam grid search
        hyper_grid = {"max_features": np.arange(0.05, 0.55, 0.05)}
        self.grid_search = GridSearchCV(
            rf,
            hyper_grid,
            scoring=score_metrics,
            cv=RepeatedGroupKFold(
                n_splits=rgk_splits, n_repeats=rgk_repeats, random_state=seed
            ),
            refit=False,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        # perform hyperparam search
        self.grid_search.fit(self.X, self.y, groups=self.groups)

    def pxls_hypertune(
        self,
        segment_id,
        gs_max_segments=50,
        rgk_splits=5,
        rgk_repeats=2,
        n_jobs=-1,
        seed=42,
        verbose=False,
    ):
        """
        Hypertunes rf based on pixel values with post-aggregation per segment,
        Computationally expensive and memory intense,
        subsetting gets very slow for more than 10 million pixels
        """
        # subsampling
        self._pxls_subsetting(segment_id, gs_max_segments, seed, verbose)
        # set up rf
        rf = RandomForestClassifier(
            n_estimators=100, random_state=seed, n_jobs=n_jobs, class_weight="balanced"
        )
        # create multiple scoring metrics
        score_metrics = {
            "pxls_acc": "accuracy",
            "pxls_f1": "f1",
            "obj_acc": make_scorer(self.obj_acc_II, metric="acc"),
            "obj_f1": make_scorer(self.obj_acc_II, metric="f1"),
        }
        # set up hyperparam grid search
        hyper_grid = {"max_features": np.arange(0.05, 0.55, 0.05)}
        self.grid_search = GridSearchCV(
            rf,
            hyper_grid,
            scoring=score_metrics,
            cv=RepeatedGroupKFold(
                n_splits=rgk_splits, n_repeats=rgk_repeats, random_state=seed
            ),
            refit=False,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        # perform hyperparam search
        self.grid_search.fit(self.X, self.y, groups=self.groups)

    def train(
        self,
        segment_id,
        max_segments=10000,
        n_jobs=-1,
        seed=42,
        verbose=False,
    ):
        """
        Train a random forest classifier pixel-wise (suitable for post-aggregation per segment),
        Group k-fold cv wrapper is used to get an unbiased estimate of the performance
        """
        # subsampling
        self._pxls_subsetting(segment_id, max_segments, seed, verbose)
        # set up rf
        rf = RandomForestClassifier(
            n_estimators=100, random_state=seed, n_jobs=n_jobs, class_weight="balanced"
        )
        # create multiple scoring metrics
        score_metrics = {
            "pxls_acc": "accuracy",
            "pxls_f1": "f1",
            "obj_acc": make_scorer(self.obj_acc_II, metric="acc"),
            "obj_f1": make_scorer(self.obj_acc_II, metric="f1"),
        }
        # set up cross-validation
        self.cv = GridSearchCV(
            rf,
            {"max_features": [0.05]},
            scoring=score_metrics,
            cv=RepeatedGroupKFold(n_splits=5, n_repeats=1, random_state=seed),
            refit="obj_acc",
            n_jobs=n_jobs,
            verbose=verbose,
        )
        # perform training
        self.cv.fit(self.X, self.y, groups=self.groups)
        self.model = self.cv.best_estimator_

    def predict(self, X, segments):
        """
        Apply trained rf to given feature vector & segment mapping
        """
        # perform prediction pixelwise
        preds_pxls = self.model.predict(X)
        # aggregate prediciton per segment
        preds_seg = pd.DataFrame({"segment_id": segments, "preds": preds_pxls})
        preds_seg = preds_seg.groupby("segment_id", sort=False).mean()
        preds_seg = preds_seg > 0.5
        self.field_ids = preds_seg.index[preds_seg.values.flatten()]
        # return segmentwise prediction & its pixelwise mapping
        self.field_pxls = np.array(segments.isin(self.field_ids))

    def obj_acc_I(self, y_true, y_pred, metric="acc"):
        """
        Calculates the accuracy on a pixel basis by weighting predictions
        for segments by their area
        """
        # calculate weight as segment's area
        tile_idxs = y_true.index.values
        w = self.weights.reindex(tile_idxs)
        w = w[w.index.isin(tile_idxs)]
        score = self.get_weighted_score(metric, w, y_true, y_pred)
        return score

    def obj_acc_II(self, y_true, y_pred, metric="acc"):
        """
        Calculates the accuracy on a pixel basis by first aggregating
        pixelwise predictions per segment and classifying it via majority vote,
        then weighting predictions for segments by their area
        """
        # compile information on segment correspondance
        tile_idxs = y_true.index.values
        gr = self.segment_id.reindex(tile_idxs)
        gr = gr[gr.index.isin(tile_idxs)]
        pxls_df = pd.DataFrame(
            {
                "segment": gr,
                "gt": y_true,
                "pred": np.array(y_pred),
            }
        )
        # get aggregated decision for each segment
        # in case of ties decide for no-field
        # future development: assess threshold as hyperparameter
        obj_df = pxls_df.groupby("segment")
        mode_gt = obj_df["gt"].mean() > 0.5
        mode_pred = obj_df["pred"].mean() > 0.5
        w = obj_df["pred"].size()
        score = self.get_weighted_score(metric, w, mode_gt, mode_pred)
        return score

    def _pxls_subsetting(self, segment_id, max_samples, seed=42, verbose=False):
        """
        Choosing at most max_samples in a random manner for
        training the pixelwise rf classifer
        """
        # subsampling stage
        np.random.seed(seed)
        comp_dict = {
            "segment_id": segment_id,
            "tile_id": self.groups,
            "gt_label": self.y,
        }
        df_comp = pd.DataFrame(comp_dict)
        df_comp = pd.merge(df_comp, self.X, left_index=True, right_index=True)
        ids = df_comp["segment_id"].unique()
        ids = np.random.permutation(ids)
        df_comp = df_comp.set_index("segment_id").loc[ids].reset_index()
        subset_ids = pd.unique(df_comp["segment_id"])[:max_samples]
        subset_df = df_comp[df_comp["segment_id"].isin(subset_ids)]
        uids = [
            f"{x}_{y}" for x, y in zip(subset_df["tile_id"], np.arange(len(subset_df)))
        ]
        subset_df.insert(0, "uid", uids)
        subset_df.set_index("uid", inplace=True)
        # get rf inputs
        self.X = subset_df.iloc[:, 3:]
        self.y = subset_df["gt_label"]
        self.groups = subset_df["tile_id"].values
        self.segment_id = subset_df["segment_id"]
        # info
        if verbose:
            print(f"Subsetting to {max_samples} segments.")
            print(f"Corresponds to {len(self.y)} samples (i.e. pixels).")
            print(f"Samples drawn from {len(np.unique(self.groups))} tiles.\n")

    @staticmethod
    def get_weighted_score(metric, weigths, y_true, y_pred):
        w = weigths
        if metric == "acc":
            score = accuracy_score(y_true, y_pred, sample_weight=w)
        if metric == "f1":
            score = f1_score(y_true, y_pred, sample_weight=w, zero_division=0)
        return score
