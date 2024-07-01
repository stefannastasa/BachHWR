import editdistance
import torch
from torchmetrics import Metric

from model.LabelParser import LabelParser
class CharacterErrorRate(Metric):

    def __init__(self, label_encoder: LabelParser):
        super().__init__()
        self.label_encoder = label_encoder

        self.add_state("cer_sum", default=torch.zeros(1, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("nr_samples", default=torch.zeros(1, dtype=torch.int64), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.ndim == target.ndim

        for i, (p, t) in enumerate(zip(preds, target)):
            p_str, t_str = map(tensor_to_str, (p, t))
            editd = editdistance.eval(p_str, t_str)

            self.cer_sum += editd / t.numel()
            self.nr_samples += 1

    def compute(self) -> torch.Tensor:
        return self.cer_sum / self.nr_samples.float()
