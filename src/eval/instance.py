import json
import numpy as np
import pandas as pd
import os
import torch  # necessary for COCOeval_fast
import xarray as xr
from fast_coco_eval import COCOeval_fast
from pycocotools import mask
from pycocotools.coco import COCO
from tqdm import tqdm
from utils.common import suppress_stdout


def single_gt_annot(arr, image_name, image_id, start_id=1):
    """
    Creates Coco-compatible annotations for a single 2D ground truth array,
    assumes a categorial mask input with all zero values represented by np.nan,
    assigns the same category to all objects (assuming only fields are present)
    """

    annotations = []

    unique_vals = np.unique(arr[~np.isnan(arr)])
    masks = np.stack([arr == val for val in unique_vals], axis=0)

    for i, mask_array in enumerate(masks):
        rle = mask.encode(np.asfortranarray(mask_array.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("utf-8")

        annotation = {
            "id": start_id + i,
            "image_id": image_id,
            "category_id": 1,
            "segmentation": rle,
            "score": 1.0,
            "area": int(mask_array.sum()),
            "bbox": list(mask.toBbox(rle)),
            "iscrowd": 0,
        }

        annotations.append(annotation)

    full_json = {
        "images": [
            {
                "id": image_id,
                "file_name": f"{image_name}_S2_10m_256.nc",
                "height": arr.shape[0],
                "width": arr.shape[1],
            }
        ],
        "categories": [{"id": 1, "name": "field"}],
        "annotations": annotations,
    }

    return full_json, start_id + len(masks)


def all_gt_annot(gt_paths, gt_ids, out_path, verbose=True):
    """
    Reads provided ground truth arrays, converts their bboxs & masks to COCO-Json format
    """
    # intialise summarising json for all tiles
    combined_gt_data = {
        "images": [],
        "categories": [{"id": 1, "name": "field"}],
        "annotations": [],
    }
    start_id = 1

    for img_id, label_path in tqdm(zip(gt_ids, gt_paths), disable=not verbose):
        # get annotation for tile
        label_ds = xr.open_dataset(label_path)
        label_name = os.path.split(label_path)[-1].split("_S2label_10m_256.tif")[0]
        proj_wkt = label_ds["spatial_ref"].attrs["spatial_ref"]
        label_ds.rio.write_crs(proj_wkt, inplace=True)
        field_enum = np.array(label_ds["band_data"][3, :, :])
        # convert array format to COCO json format
        gt_annots, start_id = single_gt_annot(field_enum, label_name, img_id, start_id)
        # append to summarising json
        combined_gt_data["images"].extend(gt_annots["images"])
        combined_gt_data["annotations"].extend(gt_annots["annotations"])

    # save json
    if os.path.isfile(out_path):
        raise OSError(f"File {out_path} already exists!")
    with open(out_path, "w") as gt_file:
        json.dump(combined_gt_data, gt_file)


def create_pred_annot(arr, image_id):
    """
    Creates Coco-compatible annotations for a single 2D prediction array describing labelled segments,
    assumes a categorial mask input with all zero values represented by np.nan,
    assigns the same category to all objects & confidence score based on area of objects
    """
    annotations = []

    unique_vals = np.unique(arr[~np.isnan(arr)])

    if len(unique_vals):
        masks = np.stack([arr == val for val in unique_vals], axis=0)

        for i, mask_array in enumerate(masks):
            rle = mask.encode(np.asfortranarray(mask_array.astype(np.uint8)))
            rle["counts"] = rle["counts"].decode("utf-8")

            annotation = {
                "image_id": image_id,
                "category_id": 1,
                "bbox": list(mask.toBbox(rle)),
                "segmentation": rle,
                "score": mask_array.sum() / (256 * 256),
            }

            annotations.append(annotation)

        return annotations

    else:
        pass


def run_coco(gt_json, pred_json, img_ids="all", verbose=True):
    """
    Runs instance evaluation & returns relevant metrics in dataframe format
    gt_json: ground truth json annotations
    pred_json: prediction json annotations
    img_ids: image ids to restrict the evaluation to (list format)
    """
    with suppress_stdout(verbose):
        # load ground truth and predicted segmentation data from the JSON files
        gt_coco = COCO(gt_json)
        pred_coco = gt_coco.loadRes(pred_json)

        # initialize the COCO evaluation object
        coco_eval = COCOeval_fast(gt_coco, pred_coco, "segm")

        # set number of max objects & sizes to be evaluated
        max_size = 256 * 256
        coco_eval.params.areaRng = [
            [0, max_size],
            [0 ** 2, 100],
            [100, 250],
            [250, max_size],
        ]
        coco_eval.params.maxDets = [
            1,
            10,
            max_size,
        ]
        coco_eval.params.catIds = [1]  # just to be sure

        # subset tiles to be evaluated
        if img_ids != "all":
            coco_eval.params.imgIds = img_ids

        # evaluate the results
        coco_eval.evaluate()
        coco_eval.accumulate()
        _ = coco_eval.summarize()

        # return relevant metrics as df
        index = [
            "AP_IoU_0.50:0.95",
            "AP_IoU_0.50",
            "AP_IoU_0.75",
            "AP_IoU_s",
            "AP_IoU_m",
            "AP_IoU_l",
            "AR_1",
            "AR_10",
            "AR_max",
            "AR_max_s",
            "AR_max_m",
            "AR_max_l",
        ]

        coco_stats = pd.Series(coco_eval.stats, index).drop(["AR_1", "AR_10"])

    return coco_stats
