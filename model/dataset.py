from pathlib import Path

from typing import Dict, Optional, Union, Sequence, Tuple
import torch
import numpy as np
import pandas as pd
from pyarrow import Tensor
import html
from model.LabelParser import LabelParser
from model.ImageTransforms import ImageTransforms
from model.utils import read_xml
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
import cv2 as cv

class IAMDataset(Dataset):
    MEAN = 0.8275
    STD = 0.2314
    MAX_FORM_HEIGHT = 3542
    MAX_FORM_WIDTH = 2479

    MAX_SEQ_LENS = {
        "word": 55,
        "line": 90,
        "form": 700,
    }  # based on the maximum seq lengths found in the dataset

    root: Path
    data: pd.DataFrame
    label_enc: LabelParser
    parse_method: str
    only_lowercase: bool
    transforms: Optional[A.Compose]
    id_to_idx: Dict[str, int]
    _split: str
    _return_writer_id: Optional[bool]

    def __init__(
            self,
            root: Union[Path, str],
            parse_method: str,
            split: str,
            return_writer_id: bool = False,
            only_lowercase: bool = False,
            label_enc: Optional[LabelParser] = None,
    ):
        super().__init__()
        _parse_methods = ["form", "line", "word"]
        err_message = (
            f"{parse_method} is not a possible parsing method: {_parse_methods}"
        )
        assert parse_method in _parse_methods, err_message

        _splits = ["train", "test"]
        err_message = f"{split} is not a possible split: {_splits}"
        assert split in _splits, err_message

        self._split = split
        self._return_writer_id = return_writer_id
        self.only_lowercase = only_lowercase
        self.root = Path(root)
        self.label_enc = label_enc
        self.parse_method = parse_method

        # Process the data.
        if not hasattr(self, "data"):
            self.data = self._get_forms()

        # Create the label encoder.
        if self.label_enc is None:
            vocab = []
            s = "".join(self.data["target"].tolist())

            if self.only_lowercase:
                s = s.lower()
            vocab += sorted(list(set(s)))
            self.label_enc = LabelParser().fit(vocab)
        if not "target_enc" in self.data.columns:
            self.data.insert(
                2,
                "target_enc",
                self.data["target"].apply(
                    lambda s: np.array(
                        self.label_enc.encode_labels(
                            [c for c in (s.lower() if self.only_lowercase else s)]
                        )
                    )
                ),
            )

        self.transforms = self._get_transforms(split)
        self.id_to_idx = {
            Path(self.data.iloc[i]["img_path"]).stem: i for i in range(len(self))
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data.iloc[idx]
        img = cv.imread(data["img_path"], cv.IMREAD_GRAYSCALE)
        if all(col in data.keys() for col in ["bb_y_start", "bb_y_end"]):
            # Crop the image vertically.
            img = img[data["bb_y_start"]: data["bb_y_end"], :]
        assert isinstance(img, np.ndarray), (
            f"Error: image at path {data['img_path']} is not properly loaded. "
            f"Is there something wrong with this image?"
        )
        if self.transforms is not None:
            img = self.transforms(image=img)["image"]
        if self._return_writer_id:
            return img, data["writer_id"], data["target_enc"]
        return img, data["target_enc"]

    def get_max_height(self):
        return (self.data["bb_y_end"] - self.data["bb_y_start"]).max()

    @property
    def vocab(self):
        return self.label_enc.classes

    @staticmethod
    def collate_fn(
            batch: Sequence[Tuple[np.ndarray, np.ndarray]],
            dataset_returns_writer_id: bool = False,
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
        if dataset_returns_writer_id:
            imgs, writer_ids, targets = zip(*batch)
        else:
            imgs, targets = zip(*batch)

        img_sizes = [im.shape for im in imgs]
        if (
                not len(set(img_sizes)) == 1
        ):  # images are of varying sizes, so pad them to the maximum size in the batch
            hs, ws = zip(*img_sizes)
            pad_fn = A.PadIfNeeded(
                max(hs), max(ws), border_mode=cv.BORDER_CONSTANT, value=0
            )
            imgs = [pad_fn(image=im)["image"] for im in imgs]
        imgs = np.stack(imgs, axis=0)
        imgs, targets = torch.Tensor(imgs), torch.Tensor(targets)
        return imgs, targets

    def set_transforms_for_split(self, split: str):
        _splits = ["train", "val", "test"]
        err_message = f"{split} is not a possible split: {_splits}"
        assert split in _splits, err_message
        self.transforms = self._get_transforms(split)

    def _get_transforms(self, split: str) -> A.Compose:
        max_img_w = self.MAX_FORM_WIDTH

        if self.parse_method == "form":
            max_img_h = (self.data["bb_y_end"] - self.data["bb_y_start"]).max()
        else:  # word or line
            max_img_h = self.MAX_FORM_HEIGHT

        transforms = ImageTransforms(
            (max_img_h, max_img_w), (IAMDataset.MEAN, IAMDataset.STD)
        )

        if split == "train":
            return transforms.train_trnsf
        elif split == "test" or split == "val":
            return transforms.test_trnsf

    def statistics(self) -> Dict[str, float]:
        assert len(self) > 0
        tmp = self.transforms
        self.transforms = None
        mean, std, cnt = 0, 0, 0
        for img, _ in self:
            mean += np.mean(img)
            std += np.var(img)
            cnt += 1
        mean /= cnt
        std = np.sqrt(std / cnt)
        self.transforms = tmp
        return {"mean": mean, "std": std}

    def get_max_target_len(self):
        return (self.data["target_len"]).max()

    def _get_forms(self) -> pd.DataFrame:

        data = {
            "img_path": [],
            "img_id": [],
            "target": [],
            "bb_y_start": [],
            "bb_y_end": [],
            "target_len": [],
        }
        for form_dir in ["formsA-D", "formsE-H", "formsI-Z"]:
            dr = self.root / form_dir
            for img_path in dr.iterdir():
                doc_id = img_path.stem
                xml_root = read_xml(self.root / "xml" / (doc_id + ".xml"))

                bb_y_start = int(xml_root[1][0].get("asy")) - 150
                bb_y_end = int(xml_root[1][-1].get("dsy")) + 150

                form_text = []
                for line in xml_root.iter("line"):
                    form_text.append(html.unescape(line.get("text", "")))

                img_w, img_h = Image.open(str(img_path)).size
                target = " ".join(form_text)
                target = target.replace("\n", " ")
                data["img_path"].append(str(img_path))
                data["img_id"].append(doc_id)
                data["target"].append(target)
                data["bb_y_start"].append(bb_y_start)
                data["bb_y_end"].append(bb_y_end)
                data["target_len"].append(len(target))
        return pd.DataFrame(data).sort_values(
            "target_len"
        )  # by default, sort by target length
