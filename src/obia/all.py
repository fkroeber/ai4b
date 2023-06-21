import gc
import os
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
from functools import partial
from skopt import gp_minimize
from skimage.morphology import remove_small_objects

from eval.all import calc_metrics
from obia.cws import CannyWater
from obia.rf import FeatEngineer, RFClassifier
from utils.common import compress_pickle, chunks


class OBIAPipeline:
    """
    Patches several components of OBIA pipeline together
    """

    def __init__(
        self,
        train_samples,
        test_samples,
        save_dir,
        summary_tiles_path,
        data_dir="d:/thesis/data/ai4boundaries/filtered",
        n_cores=os.cpu_count() - 1,
        batch_size=50,
        verbose=True,
    ):
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.summary_tiles = summary_tiles_path
        self.verbose = verbose
        self.batch_size = batch_size
        self.n_cores = n_cores
        self.bar = None  # progress bar callback

        # create folders for saving results
        self.save_pred_dir = os.path.join(self.save_dir, "test_preds")
        self.save_eval_dir = os.path.join(self.save_dir, "test_eval")
        os.makedirs(self.save_dir, exist_ok=False)
        os.makedirs(self.save_pred_dir)
        os.makedirs(self.save_eval_dir)

    def bayes_optim(self, search_space, n_calls=100, n_initial=64):
        """
        Runs bayesian optimisation for segmentation,
        incl. assessment on training samples
        """
        # iterative optimisation
        tqdm_desc = f"Bayes optim for {len(self.train_samples)} tiles"
        with tqdm(total=n_calls, desc=tqdm_desc, disable=not self.verbose) as self.bar:
            res = gp_minimize(
                partial(
                    CannyWater.obj_fun_parallel,
                    self.train_samples,
                    data_dir=self.data_dir,
                    n_cores=self.n_cores,
                ),
                search_space,
                acq_func="EI",
                acq_optimizer="sampling",
                initial_point_generator="sobol",
                n_calls=n_calls,
                n_initial_points=n_initial,
                callback=self._callback_tqdm,
                random_state=42,
            )
        # remove irrelevant parts of optim res
        res["specs"]["args"]["callback"] = None
        res.pop("random_state")
        res.pop("models")
        # evaluate results on train set
        metrics = ["f1", "f1_w", "iou", "iou_boundary"]
        res_eval = CannyWater.assess_parallel(
            samples=self.train_samples,
            hyperparams=res["x"],
            metrics=metrics,
            data_dir=self.data_dir,
            agg=True,
            n_cores=self.n_cores,
            verbose=self.verbose,
        )
        samples = [x[0] for x in res_eval]
        segment_m = [x[1].stats_agg for x in res_eval]
        res_eval = pd.DataFrame(segment_m, index=samples)
        # write results
        self.bayes_res = res
        self.bayes_eval = res_eval
        compress_pickle(os.path.join(self.save_dir, "bayes_optim.pbz2"), self.bayes_res)
        compress_pickle(os.path.join(self.save_dir, "bayes_eval.pbz2"), self.bayes_eval)

    def rf_train(self, max_segments=25000):
        """
        Trains the random forest classifier for binary classification of segments,
        incl. assessment on validation samples (splitting train samples for cv)
        """
        # engineering features
        feats = FeatEngineer.eng_parallel(
            samples=self.train_samples,
            cws_params=self.bayes_res["x"],
            feat_set=["spec", "text"],
            feat_type="pxls_feats",
            env_covars=True,
            only_feats=True,
            data_dir=self.data_dir,
            stats_csv=self.summary_tiles,
            n_cores=self.n_cores,
            verbose=self.verbose,
        )
        # extract rf inputs
        X = feats.iloc[:, 3:]
        y = feats["gt_label"]
        groups = feats["tile"].values
        segments_uids = [f"{x}_{y}" for x, y in zip(feats["tile"], feats["segment_id"])]
        segments_uids = pd.Series(segments_uids, index=y.index)
        # training
        rf_class = RFClassifier(X, y, groups)
        rf_class.train(
            segments_uids,
            n_jobs=self.n_cores,
            max_segments=max_segments,
            verbose=self.verbose,
        )
        # evaluate results on train set
        acc_cols = [
            "mean_test_obj_acc",
            "mean_test_obj_f1",
            "mean_test_pxls_acc",
            "mean_test_pxls_f1",
        ]
        rf_eval = pd.DataFrame(rf_class.cv.cv_results_)[acc_cols]
        self.rf_eval = rf_eval.T.squeeze()
        # write results
        self.rf_model = rf_class.model
        compress_pickle(os.path.join(self.save_dir, "rf_model.pbz2"), self.rf_model)
        compress_pickle(os.path.join(self.save_dir, "rf_eval.pbz2"), self.rf_eval)

    def predict(self):
        """
        Runs trained model on test sample set,
        outputs binary masks for each tile
        """
        # get batched test samples
        batched_samples = list(chunks(self.test_samples, self.batch_size))

        # run pipeline in batched manner to avoid memory buffer overflow
        tqdm_desc = f"Process {len(batched_samples)} test sample batches"
        for batch in tqdm(batched_samples, disable=not self.verbose, desc=tqdm_desc):
            # cws segmentation & feat engineering
            cws_feats = FeatEngineer.eng_parallel(
                samples=batch,
                cws_params=self.bayes_res["x"],
                feat_set=["spec", "text"],
                feat_type="pxls_feats",
                env_covars=True,
                only_feats=False,
                data_dir=self.data_dir,
                stats_csv=self.summary_tiles,
                n_cores=self.n_cores,
                verbose=False,
            )
            # extract rf inputs
            feats = pd.concat([x.feats_df for x in cws_feats])
            X = feats.iloc[:, 3:]
            y = feats["gt_label"]
            groups = feats["tile"].values
            zipped_tile_ids = zip(feats["tile"], feats["segment_id"])
            segments_uids = [f"{x}_{y}" for x, y in zipped_tile_ids]
            segments_uids = pd.Series(segments_uids, index=y.index)
            # apply rf model
            rf = RFClassifier(X, y, groups)
            rf.model = self.rf_model
            rf.predict(X, segments_uids)
            # get index limits for stacked pixelwise predicitions
            idx_limits_lower = np.linspace(0, len(rf.field_pxls), len(cws_feats) + 1)
            idx_limits_upper = np.roll(idx_limits_lower, -1)
            idx_limits = [[x, y] for x, y in zip(idx_limits_lower, idx_limits_upper)][
                :-1
            ]
            # individual tile processing
            for (idx_I, idx_II), cws_feat in zip(idx_limits, cws_feats):
                tile = cws_feat.cws.tile_name
                # extract valid segments classified as cropland
                # note min idx for gt & pred is 1 -> thus writing to idx 0 or negative safe
                fields = rf.field_pxls[int(idx_I) : int(idx_II)].reshape(256, 256)
                fields = np.where(fields, cws_feat.cws.segments, 0).astype(np.int32)
                # remove small objects
                fields = remove_small_objects(fields, min_size=25)
                fields = np.where(fields > 0, fields, -1)
                # write results to disk
                fields_georef = xr.DataArray(
                    fields.astype(np.int32),
                    coords={"y": cws_feat.cws.tile_ds.y, "x": cws_feat.cws.tile_ds.x},
                    dims=["y", "x"],
                )
                dst = os.path.join(self.save_pred_dir, f"{tile}.tif")
                fields_georef.rio.to_raster(dst, compress="LZW")
            # clean up
            del cws_feats, feats
            gc.collect()

    def eval(self, gt_json_path):
        """
        Runs model evaluation on test sample set,
        outputs three kinds of metrics
            + m1: classification metrics (pixelwise accuracy of cropland mask)
            + m2: segmentation metrics (raster & object-based focused on boundaries)
            + m3: instance segmentation metrics (mask-based focus on fields)
        """
        calc_metrics(
            preds_save_dir=self.save_pred_dir,
            gt_data_dir=self.data_dir,
            gt_json_path=gt_json_path,
            eval_save_dir=self.save_eval_dir,
            summary_stats_path=self.summary_tiles,
            verbose=self.verbose,
        )

    def _callback_tqdm(self, x):
        self.bar.update()
