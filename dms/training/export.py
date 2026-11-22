"""Export checkpoints to ONNX (+ optional INT8 dynamic quantisation) and benchmark latency.

    python -m dms.training.export runs/face/best.pt --int8
    python -m dms.training.export --benchmark-only --backbone mobilenetv3_large_100   # untrained, latency only
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..models import checkpoint
from ..models.networks import CabinNet, FaceStateNet, build_temporal, count_params


class _FaceWrapper(torch.nn.Module):
    """ONNX needs tuple outputs, not dicts."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        o = self.m(x)
        return tuple(o[k] for k in self.m.tasks)


def example_input(kind: str, img_size: int = 224, seq_len: int = 300, in_dim: int = 9):
    return torch.randn(1, seq_len, in_dim) if kind == "temporal" else torch.randn(1, 3, img_size, img_size)


def bench_torch(model, x, n=50, warm=10) -> float:
    with torch.inference_mode():
        for _ in range(warm):
            model(x)
        t = time.perf_counter()
        for _ in range(n):
            model(x)
    return (time.perf_counter() - t) / n * 1000


def bench_onnx(path, x, n=50, warm=10, threads=2) -> float:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    feed = {s.get_inputs()[0].name: x.numpy()}
    for _ in range(warm):
        s.run(None, feed)
    t = time.perf_counter()
    for _ in range(n):
        s.run(None, feed)
    return (time.perf_counter() - t) / n * 1000


def export(model, kind, x, path: Path, int8: bool) -> list[Path]:
    m = _FaceWrapper(model) if kind == "face" else model
    torch.onnx.export(m, x, path, input_names=["input"], dynamic_axes={"input": {0: "batch"}}, opset_version=17, dynamo=False)
    paths = [path]
    if int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        q = path.with_name(path.stem + "_int8.onnx")
        # Dynamic INT8 only pays off for MatMul/Gemm. ONNX Runtime's ConvInteger
        # kernels are slower than FP32 conv on most CPUs (measured: MobileNetV3
        # 3 ms FP32 -> 52 ms when convs were included), so CNNs need *static* QDQ
        # quantisation with calibration images instead (onnxruntime.quantization.quantize_static).
        quantize_dynamic(str(path), str(q), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"])
        paths.append(q)
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", nargs="?")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--benchmark-only", action="store_true")
    ap.add_argument("--kind", default="face", choices=["face", "cabin", "temporal"])
    ap.add_argument("--backbone", default="mobilenetv3_large_100")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="runs/export")
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if a.ckpt:
        model, ck = checkpoint.load(a.ckpt)
        kind = ck["kind"]
        x = example_input(kind, ck.get("img_size", 224), ck.get("seq_len", 300), ck["arch"].get("in_dim", 9))
        name = Path(a.ckpt).parent.name + "_" + kind
    else:
        kind = a.kind
        model = {"face": lambda: FaceStateNet(a.backbone, pretrained=False), "cabin": lambda: CabinNet(a.backbone, pretrained=False), "temporal": lambda: build_temporal("transformer")}[kind]().eval()
        x = example_input(kind)
        name = f"{kind}_{a.backbone if kind != 'temporal' else 'transformer'}"

    report = {"model": name, "params_M": count_params(model) / 1e6, "threads": a.threads, "torch_ms": bench_torch(model, x)}
    for p in export(model, kind, x, out / f"{name}.onnx", a.int8):
        key = "onnx_int8" if p.stem.endswith("_int8") else "onnx_fp32"
        report[f"{key}_ms"] = bench_onnx(p, x, threads=a.threads)
        report[f"{key}_MB"] = p.stat().st_size / 1e6
    print(json.dumps(report, indent=2))
    with open(out / "benchmarks.jsonl", "a") as f:
        f.write(json.dumps(report) + "\n")


if __name__ == "__main__":
    main()
