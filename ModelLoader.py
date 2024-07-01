from torch import nn

from model.LabelParser import LabelParser
from model.model import ConvHWR, PadPool
from model.utils import pickle_load
import torch
import os
class ModelLoader:
    label_enc: LabelParser
    @staticmethod
    def _get_statedict():
        MODEL_CHECKPOINT = os.environ.get("MODEL_CHECKPOINT", "./ConvHWR.ckpt")
        checkpoint = torch.load(MODEL_CHECKPOINT,
                                map_location="cpu")

        state_dict = checkpoint["state_dict"]
        new_state_dict = {}

        for key in state_dict.keys():
            new_key = key.replace("model.", "")
            new_state_dict[new_key] = state_dict[key]

        return new_state_dict

    def __init__(self):
        LABEL_ENC = os.environ.get("LABEL_ENC", "./label_enc")

        self.label_enc = pickle_load(LABEL_ENC)

        self.model = ConvHWR(
            n_channels = 1,
            label_enc=self.label_enc,
            mul_rate= 1.0,
            layer_resizes= {
                    0: nn.MaxPool2d(2, 2),
                    1: nn.MaxPool2d(2, 2),
                    2: nn.MaxPool2d(2,2),
                    3: PadPool(),
                    6: nn.Upsample((450, 15), align_corners=True, mode="bilinear"),
                    7: nn.Upsample((1100, 8), align_corners=True, mode="bilinear")
                },
            layer_sizes={
                0: 512,
                3: 1024,
                7: 512
            },
            num_layers=8,
            fup=16,
        )

        self.model.load_state_dict(self._get_statedict(), strict=False)
        self.model.eval()

    def predict(self, input_image):
        predictions = self.model(input_image, None)

        _, val = predictions.max(dim=-1)
        val = val.squeeze()

        # val = val[val != 0]
        return self.label_enc.ctc_decode_labels(val.tolist())

