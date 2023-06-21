import pickle
import os
import random
import subprocess
import sys
import time

default_configs = {
    "project_dir": "/share/projects/ai4b",
    "save_dir": "results/instance_seg/final",
    "mm_library": "mmdetection",
    "subset_name": "final_size_1000_idx_0_seed_42",
    "lr": 0.01,
    "optimiser": "SGD",
    "scheduler": "multistep",
    "model_name": "maskrcnn",
    "backbone": "ResNet",
    "max_epochs": 50,
    "patience": 7,
    "batch_size": 2,
    "pp_type": "mask",
    "score_thres": 0.25,
    "iou_thres": 0.05,
    "n_cores": os.cpu_count() - 1,
}


class DLPipeline:
    """
    Patches several components of deep learning pipeline together
    including postprocessing & evaluation for instance segmentation results
    """

    def __init__(
        self,
        project_dir="/share/projects/ai4b",
        save_dir="results/instance_seg/final",
        mm_library="mmdetection",
        pp_type="mask",
        score_thres=0.05,
        iou_thres=0.5,
        n_cores=os.cpu_count() - 1,
        verbose=True,
    ):
        self.project_dir = project_dir
        self.save_dir = os.path.normpath(
            os.path.join(project_dir, save_dir, self.get_timestamp())
        )
        self.mm_library = mm_library
        self.n_cores = n_cores
        self.verbose = verbose
        self.pp_type = pp_type
        self.score_thres = score_thres
        self.iou_thres = iou_thres
        # create folders for saving results
        self.save_pred_dir = os.path.join(self.save_dir, "test_preds")
        self.save_eval_dir = os.path.join(self.save_dir, "test_eval")
        os.makedirs(self.save_pred_dir)
        os.makedirs(self.save_eval_dir)
        # get other paths
        self.gt_data_dir = os.path.join(
            self.project_dir, "data", "ai4boundaries", "filtered"
        )
        self.gt_annot_path = os.path.join(
            self.project_dir, "data", "ai4boundaries", "annotations.json"
        )
        self.stats_path = os.path.join(
            self.project_dir, "results", "eda", "filtered_stats_tiles.csv"
        )

    # note: configure script & underlying base configs not provided in this repo
    # confiuration for best model can be found under best_model -> oms_rcnn_config.py
    def configure(
        self,
        subset_name="final_size_100_idx_0_seed_42",
        lr=0.01,
        optimiser="SGD",
        scheduler="multistep",
        model_name="maskrcnn",
        backbone="ECAResNet",
        batch_size=2,
        max_epochs=50,
        patience=7,
    ):
        mm_abb = "mmdet" if self.mm_library == "mmdetection" else "mmrot"
        mm_config_py = os.path.join("scripts", "wrappers", f"{mm_abb}_config.py")
        mm_config_py = os.path.normpath(os.path.join(self.project_dir, mm_config_py))
        subprocess.run(
            [
                sys.executable,
                "-u",
                str(mm_config_py),
                "--save_dir",
                str(self.save_dir),
                "--subset_name",
                str(subset_name),
                "--lr",
                str(lr),
                "--optimiser",
                str(optimiser),
                "--scheduler",
                str(scheduler),
                "--model_name",
                str(model_name),
                "--backbone",
                str(backbone),
                "--batch_size",
                str(batch_size),
                "--max_epochs",
                str(max_epochs),
                "--patience",
                str(patience),
            ],
            check=True,
        )

    def train(self):
        mm_train_py = os.path.join("repos", self.mm_library, "tools", "train.py")
        mm_train_py = os.path.normpath(os.path.join(self.project_dir, mm_train_py))
        subprocess.run(
            [
                sys.executable,
                "-u",
                mm_train_py,
                os.path.join(self.save_dir, "config.py"),
            ],
            check=True,
        )
        self.clean_files(self.save_dir)

    def predict(self):
        mm_test_py = os.path.join("repos", "mmdetection", "tools", "test.py")
        mm_test_py = os.path.normpath(os.path.join(self.project_dir, mm_test_py))
        best_pth = self.find_best_pth(self.save_dir)
        iou_thres = 0.5 if self.pp_type == "mask" else self.iou_thres
        subprocess.run(
            [
                sys.executable,
                "-u",
                mm_test_py,
                os.path.join(self.save_dir, "config.py"),
                best_pth,
                "--out",
                os.path.join(self.save_dir, "preds_raw.pkl"),
                "--cfg-options",
                'model.test_cfg.rcnn.score_thr="0.05"',
                f'model.test_cfg.rcnn.nms.iou_threshold="{iou_thres}"',
            ],
            check=True,
        )

    def postprocess(self):
        """
        Transforms overlapping raw predictions into set of mutually exclusive field segments
        by performing non-maxima suppression using the polygon shapes
        """
        with open(os.path.join(self.save_dir, "preds_raw.pkl"), "rb") as file:
            preds = pickle.load(file)
        print(f"PP with params: score {self.score_thres}, IoU {self.iou_thres}")
        IPP.pp_parallel(
            preds,
            25,
            self.score_thres,
            self.iou_thres,
            pp_type=self.pp_type,
            data_dir=self.gt_data_dir,
            n_cores=self.n_cores,
            save_dir=self.save_pred_dir,
            verbose=self.verbose,
        )

    def eval(self):
        """
        Perform evaluation by calculating the standard set of metrics defined in eval
        """
        calc_metrics(
            preds_save_dir=self.save_pred_dir,
            gt_data_dir=self.gt_data_dir,
            gt_json_path=self.gt_annot_path,
            eval_save_dir=self.save_eval_dir,
            summary_stats_path=self.stats_path,
            verbose=self.verbose,
        )

    @staticmethod
    def get_timestamp():
        delay = random.randint(0, 1)
        time.sleep(delay)
        return time.strftime("%Y%m%d_%H%M%S")

    @staticmethod
    def clean_files(dir):
        for filename in os.listdir(dir):
            file_path = os.path.join(dir, filename)
            if filename.endswith(".pth") and "best_" not in filename:
                os.remove(file_path)
            elif "last_checkpoint" in filename:
                os.remove(file_path)

    @staticmethod
    def find_best_pth(dir):
        for filename in os.listdir(dir):
            if filename.startswith("best_") and filename.endswith(".pth"):
                return os.path.normpath(os.path.join(dir, filename))


if __name__ == "__main__":
    # parse arguments
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Run deep learning pipeline for instance segmentation of fields",
    )
    parser.add_argument(
        "--project_dir",
        type=str,
        default=default_configs["project_dir"],
        help="project dir to find correct paths subsequently",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=default_configs["save_dir"],
        help="directory to save results, relative to project_dir",
    )
    parser.add_argument(
        "--mm_library",
        type=str,
        choices=["mmdetection", "mmrotate"],
        default=default_configs["mm_library"],
        help="mmdetection library used for model training",
    )
    parser.add_argument(
        "--subset_name",
        type=str,
        default=default_configs["subset_name"],
        help="AI4B subset used for training & evaluation",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=default_configs["lr"],
        help="model training learning rate",
    )
    parser.add_argument(
        "--optimiser",
        type=str,
        default=default_configs["optimiser"],
        help="model training optimiser",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default=default_configs["scheduler"],
        help="model training scheduler",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=default_configs["model_name"],
        help="model name",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=default_configs["backbone"],
        help="model backbone",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=default_configs["batch_size"],
        help="batch size for model training",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=default_configs["max_epochs"],
        help="max number of epochs for model training",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=default_configs["patience"],
        help="patience for early stopping on validation set",
    )
    parser.add_argument(
        "--pp_type",
        type=str,
        choices=["mask", "bbox"],
        default=default_configs["pp_type"],
        help="postprocessing type for NMS",
    )
    parser.add_argument(
        "--score_thres",
        type=float,
        default=default_configs["score_thres"],
        help="score threshold for NMS postprocessing",
    )
    parser.add_argument(
        "--iou_thres",
        type=float,
        default=default_configs["iou_thres"],
        help="iou threshold for NMS postprocessing",
    )
    parser.add_argument(
        "--n_cores",
        type=int,
        default=default_configs["n_cores"],
        help="number of CPU cores for parallelisation",
    )

    # parse variables
    config = vars(parser.parse_args())

    # module imports
    package_dir = os.path.join(config["project_dir"], "scripts")
    sys.path.append(package_dir)
    from dl.postprocess import InstancePostProcessor as IPP
    from eval.all import calc_metrics

    # run dl pipeline
    dlp = DLPipeline(
        project_dir=config["project_dir"],
        save_dir=config["save_dir"],
        mm_library=config["mm_library"],
        pp_type=config["pp_type"],
        score_thres=config["score_thres"],
        iou_thres=config["iou_thres"],
        n_cores=config["n_cores"],
    )
    dlp.configure(
        subset_name=config["subset_name"],
        lr=config["lr"],
        optimiser=config["optimiser"],
        scheduler=config["scheduler"],
        model_name=config["model_name"],
        backbone=config["backbone"],
        batch_size=config["batch_size"],
        max_epochs=config["max_epochs"],
        patience=config["patience"],
    )
    dlp.train()
    dlp.predict()
    dlp.postprocess()
    dlp.eval()
