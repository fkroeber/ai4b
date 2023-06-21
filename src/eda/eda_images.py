# describe image tiles via deep feature clustering #

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import warnings
import xarray as xr

from tqdm import tqdm
from torchvision import models
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

### define default variables ###
data_dir = "d:/thesis/data"
tile_df = pd.read_csv(
    os.path.join(
        data_dir, "ai4boundaries", "ai4boundaries_ftp_urls_sentinel2_split.csv"
    )
)
tile_df.dropna(inplace=True)
tile_df["data_paths"] = [
    os.path.join(data_dir, "ai4boundaries", "original", f"{tile}_S2_10m_256.nc")
    for tile in tile_df["file_id"]
]


default_kwargs = {
    "path_imgs": tile_df["data_paths"],
    "path_save": "d:/thesis/results/eda/test.csv",
    "net": "resnet",
    "reduce_alg": "tsne",
    "img_mode": "true_color",
    "n_components": 2,
    "topo_balance": 30,
}


# define relevant classes
class ImgFeatureVisualiser:
    """
    Extracts deep features & applies dimensionality reduction to enable 2D visualisations
    for a multitemporal dataset (specifically AI4boundaries)
    """

    def __init__(self, **kwargs):
        """
        path_imgs: list with paths to netcdf imgs
        path_save: path to save final dataframe
        net: net backbone to be used for feat extraction (resnet or vit)
        reduce_alg: dimensionality reduction algorithm to be used (tsne or umap)
        img_mode: "true_color" (i.e. r,g,b) or "false_color" (i.e. nir,r,g)
        n_components: number of components to be extracted, usually either 2 or 3
        topo_balance: greater vals, give more weight to global structure in dataset
        """
        # initialise arguments
        self.config = kwargs
        self.path_imgs = kwargs.get("path_imgs")
        self.path_save = kwargs.get("path_save")
        self.net = kwargs.get("net")
        self.reduce_alg = kwargs.get("reduce_alg")
        self.img_mode = kwargs.get("img_mode")
        self.n_components = kwargs.get("n_components")
        self.topo_balance = kwargs.get("topo_balance")
        os.makedirs(os.path.dirname(self.path_save), exist_ok=True)

    def feat_extract(self):
        """
        Extract image features via pretrained CNN or Transformer
        """
        # define identity to modify pretrained nets as feature extractors
        class Identity(nn.Module):
            def __init__(self):
                super(Identity, self).__init__()

            def forward(self, x):
                return x

        # define data set specific image class
        # re-arrage data into temporal chunks of r,g,b
        class ImageDataset(Dataset):
            def __init__(self, image_paths, composite, transform=None):
                self.image_paths = list(image_paths)
                self.composite = composite
                self.transform = transform

            def __len__(self):
                return len(self.image_paths)

            def __getitem__(self, idx):
                tile_name = os.path.split(self.image_paths[idx])[-1]
                tile_path = self.image_paths[idx]
                tile_ds = xr.open_dataset(tile_path)
                if self.composite == "true_color":
                    tile_ds = xr.merge([tile_ds["B4"], tile_ds["B3"], tile_ds["B2"]])
                elif self.composite == "false_color":
                    tile_ds = xr.merge([tile_ds["B8"], tile_ds["B4"], tile_ds["B3"]])
                tile_ds = np.array(tile_ds.to_array()).reshape(-1, 256, 256) / 10000
                rgb_idxs = np.arange(18).reshape(3, 6)
                rgb_idxs = [rgb_idxs[:, i] for i in range(rgb_idxs.shape[1])]
                rgbs = np.vstack([tile_ds[x] for x in rgb_idxs])
                img = torch.from_numpy(rgbs)
                if self.transform:
                    return self.transform(img), tile_name
                else:
                    return img, tile_name

        # choose net_backbone & modify as feature extractor
        if self.net == "resnet":
            w = models.ResNet34_Weights.DEFAULT
            model_transform = w.transforms(antialias=True)
            model = models.resnet34(weights=w)
            model.fc = Identity()
        elif self.net == "vit":
            w = models.Swin_T_Weights.DEFAULT
            model_transform = w.transforms(antialias=True)
            model = models.swin_t(weights=w)
            model.head = Identity()

        # load train & test data
        dataset = ImageDataset(self.path_imgs, self.img_mode)
        batch_size = 64
        dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # apply model
        # returns temporally stacked feature vector
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()
        with torch.no_grad(), tqdm(dl, desc="feature extraction") as pbar:
            features, img_paths = [], []
            for batch, (data, labels) in enumerate(pbar):
                features_batch = []
                for i in np.arange(0, data.shape[1], 3):
                    rgb = data[:, i : i + 3, :, :]
                    output = model(model_transform(rgb).to(device))
                    features_batch.append(output.cpu().numpy())
                features.append(np.hstack(features_batch))
                img_paths.extend(labels)
            self.features = np.vstack(features)
            self.img_paths = img_paths

    def dim_reduce(self):
        """
        Apply dimensionality reduction algorithm
        """
        if self.reduce_alg == "tsne":
            try:
                # gpu implementation of tsne
                # https://medium.com/rapids-ai/tsne-with-gpus-hours-to-seconds-9d9c17c941db
                from cuml.manifold import TSNE

                tsne_ = TSNE(
                    method="fft",
                    n_components=self.n_components,
                    perplexity=self.topo_balance,
                    n_neighbors=int(3 * self.topo_balance),
                    random_state=42,
                )
                self.X_embed = tsne_.fit_transform(self.features)
            except ModuleNotFoundError:
                # cpu implementation of tsne
                warnings.warn(
                    f"""
                GPU implementation of TSNE not available. 
                Depending on the size of the array the processing will take a while.
                The array to be processed has shape: {self.features.shape}.
                """
                )
                from sklearn.manifold import TSNE

                tsne_ = TSNE(
                    n_components=self.n_components,
                    learning_rate="auto",
                    perplexity=self.topo_balance,
                    random_state=42,
                )
                self.X_embed = tsne_.fit_transform(self.features)
        elif self.reduce_alg == "umap":
            # cpu implementation of umap
            import umap

            umap_ = umap.UMAP(
                n_components=self.n_components,
                n_neighbors=self.topo_balance,
                random_state=42,
            )
            self.X_embed = umap_.fit_transform(self.features)

    def summarise(self):
        """
        Merge metadata information with embedded image features as a dataframe
        """
        # merge embedded features with img path information
        df_embed = pd.DataFrame(
            self.X_embed, [x.split("_S2_")[0] for x in self.img_paths]
        )
        df_embed.columns = [f"x{x+1}" for x in np.arange(self.X_embed.shape[1])]
        self.df_embed = (df_embed - df_embed.min()) / (df_embed.max() - df_embed.min())
        # insert information on net type & dimensionality reduction params
        for k, v in self.config.items():
            if k in ["net", "reduce_alg", "topo_balance", "img_mode"]:
                self.df_embed[k] = v
        # save dataframe
        self.df_embed.to_csv(self.path_save)


if __name__ == "__main__":
    # test various settings for feature visualisation
    from itertools import product

    config = default_kwargs
    save_dir = "d:/thesis/results/eda/"

    img_modes = ["true_color", "false_color"]
    reduce_algs = ["tsne", "umap"]
    balances = [5, 10, 30, 50]
    hyperparams = list(product(reduce_algs, balances))

    # perform feature extraction & corresponding reduction for given settings
    i = 0
    for img_mode in img_modes:
        config["img_mode"] = img_mode
        ifv = ImgFeatureVisualiser(**config)
        ifv.feat_extract()
        for reduce_alg, balance in hyperparams:
            ifv.reduce_alg = reduce_alg
            ifv.config["reduce_alg"] = reduce_alg
            ifv.topo_balance = balance
            ifv.config["topo_balance"] = balance
            ifv.path_save = os.path.join(save_dir, f"imgfeatvis_{i}.csv")
            ifv.config["path_save"] = os.path.join(save_dir, f"imgfeatvis_{i}.csv")
            ifv.dim_reduce()
            ifv.summarise()
            i += 1
            print(f"Setting no. {i} tested.")

    # merge final results
    img_feat_df_paths = [
        os.path.join(save_dir, f"imgfeatvis_{i}.csv") for i in np.arange(i)
    ]
    img_feat_df = [pd.read_csv(path, index_col=0) for path in img_feat_df_paths]
    img_feat_df = pd.concat(img_feat_df)
    # save result & delete interim results
    img_feat_df.to_csv(
        os.path.join(save_dir, "tiles_feats.csv"), index_label="img_name"
    )
    [os.remove(x) for x in img_feat_df_paths]
