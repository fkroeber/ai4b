import json
import multiprocess
import numpy as np
import pandas as pd
import os
import torch
import torchmetrics as metrics
import xarray as xr

from eval.instance import create_pred_annot, run_coco
from eval.segment import Evaluator, m_objects
from functools import partial
from obia.cws import CannyWater
from utils.common import compress_pickle, chunks
from tqdm import tqdm


def calc_metrics(
    preds_save_dir,
    gt_data_dir,
    gt_json_path,
    eval_save_dir,
    summary_stats_path,
    verbose=True,
):
    """
    Runs model evaluation & writes three kinds of metrics to disk
        + m1: classification metrics (pixelwise accuracy of cropland mask)
        + m2: segmentation metrics (raster & object-based focused on boundaries)
        + m3: instance segmentation metrics (mask-based focus on fields)

    args:
        preds_save_dir: directory with written predictions
        gt_data_dir: directory with labelled ground truth data
        gt_json_path: path to json that contains ground truth label annotations
        eval_save_dir: directory to dave evaluation results
        summary_stats_path: path to csv that contains uids for all tiles
    """
    # get names of prediction tiles
    tiles = [x for x in os.listdir(preds_save_dir) if x[-4:] == ".tif"]

    # read tiles summary with tiles uids for instance annotations
    stats_tiles = pd.read_csv(summary_stats_path, index_col=0, header=[0, 1])
    stats_tiles.columns = stats_tiles.columns.get_level_values(1)

    # initalise accuracy metrics
    m1_collection = metrics.MetricCollection(
        [
            metrics.classification.BinaryAccuracy(),
            metrics.classification.BinaryF1Score(),
            metrics.classification.BinaryPrecision(),
            metrics.classification.BinaryRecall(),
        ]
    )
    m2_res = []
    m3_res = []

    # run segmentation metrics pipeline in multi-threaded manner
    batched_samples = list(chunks(tiles, 50))
    tqdm_desc = f"Segmentation metrics for {len(batched_samples)} batches"
    for batch in tqdm(batched_samples, disable=not verbose, desc=tqdm_desc):
        batch_res = _parallel_seg_eval(
            batch,
            preds_save_dir,
            gt_data_dir,
        )
        m2_res.append(batch_res)

    # run evaluation for each tile
    tqdm_desc = f"Classification & Instance metrics for {len(tiles)} tiles"
    for tile in tqdm(tiles, disable=not verbose, desc=tqdm_desc):
        # read tile & prediction
        tile_path = os.path.join(preds_save_dir, tile)
        tile_name = tile.split(".tif")[0]
        pred_ds = xr.open_dataset(tile_path)
        pred = np.array(pred_ds["band_data"][0])
        pred = np.where(pred == -1, np.nan, pred)
        gt_tile = CannyWater(tile_name, data_dir=gt_data_dir)
        gt_tile.read()
        gt = gt_tile.field_enum

        # evaluate classification accuracy
        binary_gt = (torch.tensor(gt) > 0).float()
        binary_pred = (torch.tensor(pred) > 0).float()
        m1_collection.update(binary_pred, binary_gt)

        # create annotations for instance accuracy evaluation
        image_id = int(stats_tiles[stats_tiles.index == tile_name]["uid"])
        pred_annotations = create_pred_annot(pred, image_id)
        if pred_annotations:
            m3_res.extend(pred_annotations)

    # compute classification metrics
    m1_res = m1_collection.compute()
    m1_res = {k: float(v) for k, v in m1_res.items()}
    m1_res = pd.Series(m1_res)
    compress_pickle(os.path.join(eval_save_dir, "m1.pbz2"), m1_res)

    # compile segmentation metrics
    m2_res = pd.concat(m2_res).reset_index(drop=True)
    compress_pickle(os.path.join(eval_save_dir, "m2.pbz2"), m2_res)

    # compute instance acc metrics
    pred_filename = os.path.join(eval_save_dir, "annots.json")
    with open(pred_filename, "w") as output_pred_file:
        json.dump(m3_res, output_pred_file)
    tiles = [x.split(".tif")[0] for x in tiles]
    img_ids = [int(stats_tiles[stats_tiles.index == x]["uid"]) for x in tiles]
    m3_res = run_coco(gt_json_path, pred_filename, img_ids, verbose=False)
    compress_pickle(os.path.join(eval_save_dir, "m3.pbz2"), m3_res)


def _seg_eval(tile_name, pred_dir, gt_dir):
    # read tile & prediction
    tile_path = os.path.join(pred_dir, tile_name)
    tile_name = tile_name.split(".tif")[0]
    pred_ds = xr.open_dataset(tile_path)
    pred = np.array(pred_ds["band_data"][0])
    pred = np.where(pred == -1, np.nan, pred)
    gt_tile = CannyWater(tile_name, data_dir=gt_dir)
    gt_tile.read()
    gt = gt_tile.field_enum

    # evaluate segmentation accuracy
    eval_m = Evaluator(pred, gt, m_objects, agg=False)
    eval_m.eval_tile()
    eval_m.stats_objects.insert(0, "tile", tile_name)
    return eval_m.stats_objects


def _parallel_seg_eval(
    tiles,
    pred_dir,
    gt_dir,
    n_cores=12,
):
    dl_eval = partial(_seg_eval, pred_dir=pred_dir, gt_dir=gt_dir)
    with multiprocess.Pool(processes=n_cores) as pool:
        feats = pool.map(dl_eval, tiles)
        pool.close()
        pool.join()
    return pd.concat(feats)
