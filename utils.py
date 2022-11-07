import os
import torch
import numpy as np
from skimage.io import imread
from collections import Counter
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

    label_counter = Counter()

    def __init__(self, image_names, mask_names, mode="train", transform=None):
        if len(image_names) != len(mask_names):
            raise ValueError("Lengths of image_names and mask_names must be same. " +\
                             f"Length of image_names = {len(image_names)}, length of mask_names = {len(mask_names)}.")
        if mode not in ["train", "val"]:
              raise NameError(f"mode must take values 'train' or 'val', got: {mode}.")
        self.images = [imread(image_name) for image_name in image_names]
        self.masks = [np.load(mask_name) for mask_name in mask_names]
        #self.class_counter = Counter()
        if mode == "train":
            for mask in self.masks:
                self.label_counter.update(mask.flatten())
        self.transform = transform or ToTensor()
        self._len = len(self.images)
        self.n_classes = len(self.label_counter)

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


def visualize(image, mask, pred_mask=None, figsize=(8, 8),
              mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    image = image.permute(1, 2, 0)
    image = image * torch.tensor(std) + torch.tensor(mean)

    plt.figure(figsize=figsize)
    if pred_mask is not None:
        plt.subplot(1, 3, 1)
        plt.title("Image")
        plt.axis("off")
        plt.imshow(image)

        plt.subplot(1, 3, 2)
        plt.title("Mask")
        plt.axis("off")
        plt.imshow(mask)

        plt.subplot(1, 3, 3)
        plt.title("Mask predicted")
        plt.axis("off")
        plt.imshow(mask)

    else:
        plt.subplot(1, 2, 1)
        plt.title("Image")
        plt.axis("off")
        plt.imshow(image)

        plt.subplot(1, 2, 2)
        plt.title("Mask")
        plt.axis("off")
        plt.imshow(mask)
    plt.show()


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