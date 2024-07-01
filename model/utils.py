import math
import pickle
import xml.etree.ElementTree as ET
from pathlib import Path
from random import random
from typing import Any, Tuple, Union

import numpy as np
import torch
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import TQDMProgressBar
from torch import Tensor
import cv2 as cv

from model.LabelParser import LabelParser


def pickle_load(file) -> Any:
    with open(file, "rb") as f:
        return pickle.load(f)


def pickle_save(obj, file):
    with open(file, "wb") as f:
        pickle.dump(obj, f)


def read_xml(file: Union[Path, str]) -> ET.Element:
    tree = ET.parse(file)
    root = tree.getroot()

    return root


def find_child_by_tag(element: ET.Element, tag: str, value: str) -> Union[ET.Element, None]:
    for child in element:
        if child.get(tag) == value:
            return child
    return None


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dpi_adjusting(img: np.ndarray, scale: float, **kwargs) -> np.ndarray:
    height, width = img.shape[:2]
    new_height, new_width = math.ceil(height * scale), math.ceil(width * scale)
    return cv.resize(img, (new_width, new_height))


class LitProgressBar(TQDMProgressBar):
    def get_metrics(self, trainer, model):
        # don't show the version number
        items = super().get_metrics(trainer, model)
        for k in list(items.keys()):
            if k.startswith("grad"):
                items.pop(k, None)
        items.pop("v_num", None)
        return items


def decode_prediction_and_target(
        pred: Tensor, target: Tensor, label_encoder: LabelParser
) -> Tuple[str, str]:
    # Decode prediction and target.
    p, t = pred.tolist(), target.tolist()
    pred_str = "".join(label_encoder.ctc_decode_labels(p))
    target_str = "".join(label_encoder.ctc_decode_labels(t))
    return pred_str, target_str


def matplotlib_imshow(
        img: torch.Tensor, mean: float = 0.5, std: float = 0.5, one_channel=True
):
    assert img.device.type == "cpu"
    if one_channel and img.ndim == 3:
        img = img.mean(dim=0)
    img = img * std + mean  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
