"""Neural networks.

FaceStateNet   one shared backbone, several heads (eye state, yawn, emotion).
               Trained on a *union* of datasets where each image carries only
               some labels (MRL-Eye -> eye, YawDD -> yawn, AffectNet/FER -> emotion);
               missing labels are -1 and masked out of the loss.
CabinNet       driver-behaviour classifier on the cabin/body view (phone, texting,
               drinking, reaching back, ...), e.g. State Farm / 100-Driver classes.
DrowsinessTransformer / DrowsinessGRU
               temporal models over the per-frame feature sequence -> alertness
               level (alert / low-vigilant / drowsy, as in UTA-RLDD).
"""
from __future__ import annotations

import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

EYE_CLASSES = ("closed", "open")
YAWN_CLASSES = ("no_yawn", "yawn")
EMOTION_CLASSES = ("neutral", "happy", "sad", "surprise", "fear", "disgust", "anger")
CABIN_CLASSES = (
    "safe_driving",
    "texting_right",
    "phone_right",
    "texting_left",
    "phone_left",
    "operating_radio",
    "drinking",
    "reaching_behind",
    "hair_makeup",
    "talking_to_passenger",
)
ALERTNESS_CLASSES = ("alert", "low_vigilant", "drowsy")

FACE_TASKS = {"eye": len(EYE_CLASSES), "yawn": len(YAWN_CLASSES), "emotion": len(EMOTION_CLASSES)}


class FaceStateNet(nn.Module):
    def __init__(self, backbone: str = "mobilenetv3_large_100", pretrained: bool = True, tasks: dict | None = None, drop: float = 0.2):
        super().__init__()
        self.tasks = dict(tasks or FACE_TASKS)
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        # Pooled-feature width; MobileNetV3/EfficientNet-style models expose a
        # separate head_hidden_size that differs from num_features.
        d = getattr(self.backbone, "head_hidden_size", None) or self.backbone.num_features
        self.heads = nn.ModuleDict({t: nn.Sequential(nn.Dropout(drop), nn.Linear(d, n)) for t, n in self.tasks.items()})
        # Learnable homoscedastic-uncertainty task weights (Kendall et al., CVPR 2018).
        self.log_vars = nn.Parameter(torch.zeros(len(self.tasks)))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.backbone(x)
        return {t: h(z) for t, h in self.heads.items()}

    def loss(self, out: dict[str, torch.Tensor], targets: dict[str, torch.Tensor], label_smoothing: float = 0.1):
        """Masked multi-task loss. targets[t] == -1 means 'no label for task t'."""
        total = out[next(iter(out))].new_zeros(())
        parts = {}
        for i, t in enumerate(self.tasks):
            y = targets[t]
            m = y >= 0
            if m.any():
                l = F.cross_entropy(out[t][m], y[m], label_smoothing=label_smoothing)
                total = total + torch.exp(-self.log_vars[i]) * l + self.log_vars[i]
                parts[t] = float(l.detach())
        return total, parts


class CabinNet(nn.Module):
    def __init__(self, backbone: str = "efficientnet_b0", pretrained: bool = True, num_classes: int = len(CABIN_CLASSES), drop: float = 0.3):
        super().__init__()
        self.net = timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes, drop_rate=drop)

    def forward(self, x):
        return self.net(x)


class DrowsinessTransformer(nn.Module):
    """Conv-Transformer hybrid over (B, T, F) feature sequences.

    A 1-D conv stem models short-range dynamics (individual blinks, ~0.1-0.5 s)
    and patchifies time by `patch` (default 5 steps = 0.5 s at 10 Hz), so a 30 s
    window becomes 60 tokens. The Transformer then models the long-range trend
    (blink-duration drift, head nods, PERCLOS build-up). ~175k parameters.
    """

    def __init__(self, in_dim: int = 9, d_model: int = 64, heads: int = 4, layers: int = 3, num_classes: int = len(ALERTNESS_CLASSES), patch: int = 5, max_len: int = 1024, drop: float = 0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_dim, d_model, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, patch, stride=patch),
        )
        self.proj = nn.LayerNorm(d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        pe = torch.zeros(max_len + 1, d_model)
        pos = torch.arange(max_len + 1).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
        layer = nn.TransformerEncoderLayer(d_model, heads, d_model * 4, drop, batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(self.stem(x.transpose(1, 2)).transpose(1, 2))
        h = torch.cat([self.cls.expand(h.shape[0], -1, -1), h], 1) + self.pe[:, : h.shape[1] + 1]
        h = self.encoder(h)
        return self.head(self.norm(h[:, 0]))


class DrowsinessGRU(nn.Module):
    """Recurrent baseline for the ablation table."""

    def __init__(self, in_dim: int = 9, hidden: int = 64, layers: int = 2, num_classes: int = len(ALERTNESS_CLASSES), drop: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, layers, batch_first=True, dropout=drop, bidirectional=True)
        self.head = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out.mean(1))


def build_temporal(kind: str, **kw) -> nn.Module:
    return {"transformer": DrowsinessTransformer, "gru": DrowsinessGRU}[kind](**kw)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
