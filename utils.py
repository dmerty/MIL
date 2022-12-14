import os
import torch
import numpy as np
import seaborn as sns
from skimage.io import imread
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from catalyst.runners import SupervisedRunner
from albumentations.pytorch import ToTensorV2 as ToTensor


def create_sample(path):
    return [str(path / file_name) for file_name in sorted(os.listdir(path))]


def extract_ids(path):
    with open(path) as file:
        ids = file.read().split("\n")
    return ids


def extract_name(full_name):
    return full_name.split("/")[-1].split(".")[0]


def split_sample(sample, train_ids, val_ids):
    names_train = [file_name for file_name in sample if extract_name(file_name) in train_ids]
    names_val = [file_name for file_name in sample if extract_name(file_name) in val_ids]
    return names_train, names_val


class PascalDataset(Dataset):
    def __init__(self, image_names, mask_names, n_classes=1, transform=None):
        if len(image_names) != len(mask_names):
            raise ValueError("Lengths of image_names and mask_names must be same. " +\
                             f"Length of image_names = {len(image_names)}, length of mask_names = {len(mask_names)}.")
        self.images = [imread(image_name) for image_name in image_names]
        self.masks = [np.load(mask_name) for mask_name in mask_names]
        self.n_classes = n_classes
        self.transform = transform or ToTensor()
        self._len = len(self.images)

    def expand_mask(self, mask):
        mask_expanded = []
        for label in range(self.n_classes):
            mask_expanded.append((mask == label).type(torch.int))
        return torch.stack(mask_expanded)

    def __len__(self):
        return self._len
  
    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        result = {"image" : image, "mask" : mask}
        result = self.transform(**result)
        result["mask_expanded"] = self.expand_mask(result["mask"])
        return result


def visualize(**images):
    n = len(images)
    plt.figure(figsize=(7, 7))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.axis("off")
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)


class PascalRunner(SupervisedRunner):
    def __init__(
        self,
        model = None,
        engine = None,
        device = "cpu",
        input_key = "features",
        output_key = "logits",
        output_mask_key = "output_mask",
        target_key = "targets",
        loss_key = "loss"
        ):
        super().__init__(
            model=model, 
            engine=engine, 
            input_key=input_key, 
            output_key=output_key, 
            target_key=target_key, 
            loss_key=loss_key)
        self._output_mask_key = output_mask_key
        self.device = device

    def forward(self, batch, **kwargs):
        output = self.model(batch[self._input_key], **kwargs)
        output = {self._output_key: output, 
                  self._output_mask_key: output.argmax(dim=1)}
        return output

    def handle_batch(self, batch):
        self.batch = {**batch, **self.forward(batch)}
        self.batch[self._target_key] = self.batch[self._target_key].type(torch.LongTensor).to(self.device)

    @torch.no_grad()
    def predict_batch(self, batch, **kwargs):
        output = self.forward(batch, **kwargs)
        return output[self._output_mask_key]

    def predict_loader(self, loader, **kwargs):
        loader = self.engine.prepare(loader)
        for batch in loader:
            yield self.predict_batch(batch, **kwargs)


def plot_metric(train_df, valid_df, metric_name, figsize=(12, 8)):
    plt.figure(figsize=figsize)
    sns.lineplot(data=valid_df, x="step", y=metric_name)
    sns.lineplot(data=train_df, x="step", y=metric_name)
    plt.title(metric_name)
    plt.legend(["valid", "train"])
    plt.grid(True)


def extract_masks(loader, mask_key="mask"):
    for batch in loader:
        yield batch[mask_key]


def extract_masks_label(masks, labels):
    for mask in masks:
        mask_expanded = []
        for label in labels:
            if isinstance(label, int):
                mask_expanded.append((mask == label).type(torch.int))
            if isinstance(label, list):
                intermediate_mask = torch.zeros_like(mask)
                for label_idx in label:
                    intermediate_mask += (mask == label_idx).type(torch.int)
                mask_expanded.append(intermediate_mask)
        yield torch.stack(mask_expanded)