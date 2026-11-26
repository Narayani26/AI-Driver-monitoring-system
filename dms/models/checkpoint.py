"""Self-describing checkpoints: weights + the constructor args needed to rebuild."""
from __future__ import annotations

from pathlib import Path

import torch

from .networks import CabinNet, FaceStateNet, build_temporal

BUILDERS = {
    "face": lambda a: FaceStateNet(pretrained=False, **a),
    "cabin": lambda a: CabinNet(pretrained=False, **a),
    "temporal": lambda a: build_temporal(**a),
    "drowsy": lambda a: CabinNet(pretrained=False, **{"num_classes": 2, **a}),
}


def save(path: str | Path, model: torch.nn.Module, kind: str, arch: dict, **extra) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": kind, "arch": arch, "state_dict": model.state_dict(), **extra}, path)


def load(path: str | Path, map_location: str = "cpu") -> tuple[torch.nn.Module, dict]:
    ck = torch.load(path, map_location=map_location, weights_only=False)
    model = BUILDERS[ck["kind"]](ck["arch"])
    model.load_state_dict(ck["state_dict"])
    return model.eval(), ck
