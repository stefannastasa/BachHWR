import math
import os
from typing import Optional

import torch
from torch import nn

from model.LabelParser import LabelParser
from model.ModelMetrics import CharacterErrorRate


def get_gpu_memory_map():
    result = os.popen('nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader').read()
    return int(result.strip())

class LayerNorm(nn.Module):
    def forward(self, x):
        return nn.functional.layer_norm(x, x.size()[1:], weight=None, bias=None, eps=1e-05)

def pCnv(inp,out,groups=1):
  return nn.Sequential(
      nn.Conv2d(inp,out,1,bias=False,groups=groups),
      nn.InstanceNorm2d(out,affine=True)
  )

def dsCnv(inp,k):
  return nn.Sequential(
      nn.Conv2d(inp,inp,k,groups=inp,bias=False,padding=(k - 1) // 2),
      nn.InstanceNorm2d(inp,affine=True)
  )

class PadPool(nn.Module):
    def forward(self, x):
        x = nn.functional.pad(x, [0, 0, 0, 1])
        x = nn.functional.max_pool2d(x,(2, 2), stride=(1, 2))
        return x

class InitBlock(nn.Module):
    def __init__(self, fup, num_channels, dropout=0.1):
        super().__init__()

        self.n1 = LayerNorm()
        self.InitSeq = nn.Sequential(
            pCnv(num_channels, fup),
            nn.Softmax(dim=1),
            dsCnv(fup, 13),
            LayerNorm(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x  = self.n1(x)
        xt = x
        x = self.InitSeq(x)
        x = torch.cat([x, xt], dim=1)
        return x

class Gate(nn.Module):
    def __init__(self, ifsz):
        super().__init__()
        self.ln = LayerNorm()

    def forward(self, x):
        t0, t1 = torch.chunk(x, 2, dim=1)
        t0 = torch.tanh(t0)
        t1.sub(2)
        t1 = torch.sigmoid(t1)

        return t1 * t0

class GateBlock(nn.Module):
    def __init__(self, ifsz, ofsz, gt = True, ksz = 3, dropout=0.1):
        super().__init__()

        cfsz = int(math.floor(ifsz / 2))
        ifsz2 = ifsz + ifsz%2

        self.sq = nn.Sequential(
            pCnv(ifsz, cfsz),
            dsCnv(cfsz, ksz),
            nn.ELU(),
            nn.Dropout(dropout),

            pCnv(cfsz, cfsz * 2),
            dsCnv(cfsz * 2, ksz),
            Gate(cfsz),
            nn.Dropout(dropout),

            pCnv(cfsz, ifsz),
            dsCnv(ifsz, ksz),
            nn.ELU(),
            nn.Dropout(dropout)
        )

        self.gt = gt

    def forward(self, x):
        y = self.sq(x)

        out = x + y
        return out

class ConvHWR(nn.Module):
    def __init__(self,
                 n_channels: int,
                 label_enc: LabelParser,
                 mul_rate,
                 layer_resizes,
                 layer_sizes,
                 num_layers,
                 fup,
                 dropout=0.1,
                 reduceAxis=3 ):
        super().__init__()

        self.layer_resizes = layer_resizes
        self.Init_sequence = InitBlock(fup, 1, dropout)
        self.label_enc = label_enc

        self.cer_metric = CharacterErrorRate(label_enc)

        layers = []
        input_size = fup + n_channels
        output_size = input_size

        for i in range(num_layers):
            output_size = int(math.floor(layer_sizes[i] * mul_rate) ) if i in layer_sizes else input_size
            layers.append(GateBlock(input_size, output_size, True, 3, dropout))

            if input_size != output_size:
                layers.append(pCnv(input_size, output_size))
                layers.append(nn.ELU())
                layers.append(nn.Dropout(dropout))
            input_size = output_size

            if i in layer_resizes:
                layers.append(layer_resizes[i])

        layers.append(LayerNorm())
        self.Gatesq = nn.Sequential(*layers)
        self.Finsq = nn.Sequential(
            pCnv(output_size, self.label_enc.vocab_size),
            nn.ELU()
        )

        self.n1 = LayerNorm()
        self.it = 0
        self.reduceAxis = reduceAxis
        self.loss_fn = nn.CTCLoss(reduction="none", zero_infinity=True)

    def forward(self, image, targets: Optional[torch.Tensor]):
        x = self.Init_sequence(image)
        x = self.Gatesq(x)
        x = self.Finsq(x)
        x = torch.mean(x, self.reduceAxis, keepdim=False)
        x = self.n1(x)
        x = x.permute(0, 2, 1)
        if targets is not None:
            logits = x
            logits = logits.permute(1, 0, 2).log_softmax(2)
            logits_size = torch.IntTensor([logits.size(0)] * targets.size(0))
            targets_size = torch.IntTensor([targets.size(1)] * targets.size(0))
            targets = targets
            torch.backends.cudnn.enabled = False
            loss = self.loss_fn(logits, targets, logits_size, targets_size).mean()
            torch.backends.cudnn.enabled = True
            return x, loss
        return x

    def calculate_metrics(self, preds: torch.Tensor, targets: torch.Tensor):
        self.cer_metric.reset()
        self.wer_metric.reset()

        cer = self.cer_metric(preds, targets)

        return {"CER": cer}
