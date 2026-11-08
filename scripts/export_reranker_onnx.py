#!/usr/bin/env python3
"""把 bge-reranker-v2-m3 导出成 int8 ONNX，专治 CPU 上的延迟预算。

为什么值得折腾
------------
重排是「每来一个问题就要跑 20 次前向」的固定开销，模型本身又是 568M 参数的 xlm-roberta。
在 CPU 上，FP32 的瓶颈是算力（本机 AVX2，没有 AVX-512）；int8 动态量化把权重压到 1/4，
再走 onnxruntime 的 VNNI/oneDNN 内核，典型收益 2~3 倍。收益多少完全看 CPU，所以必须实测，
不能照抄别人的数字——`scripts/bench_rerank.py` 就是干这个的。

导出注意
--------
- 必须开动态轴（batch、sequence_length），否则线上换个候选数就得重新导
- `truncation` 只在 tokenizer 侧做，模型只要吃定长 input_ids
- xlm-roberta 的 `type_vocab_size=1`，没有 token_type_ids 这条输入，导出时可以不管
- 量化后再验一遍相关性：int8 掉点通常很小，但「通常」不是「一定」

用法
----
    python scripts/export_reranker_onnx.py
    python scripts/export_reranker_onnx.py --model models/bge-reranker-v2-m3 \\
        --out models/bge-reranker-v2-m3-onnx-int8
    # 然后让服务走 ONNX：
    #   .env: RERANK_ONNX_PATH=./models/bge-reranker-v2-m3-onnx-int8
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="导出 int8 ONNX 重排模型")
    parser.add_argument("--model", default="D:/workbuddy-space/models/bge-reranker-v2-m3")
    parser.add_argument("--out", default="D:/workbuddy-space/models/bge-reranker-v2-m3-onnx-int8")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--no-quantize", action="store_true", help="只导出 FP32 ONNX")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = Path(args.model)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    # 分词器随 ONNX 一起放，运行时就不用再指回原始权重目录
    tokenizer.save_pretrained(out_dir / "tokenizer")

    dummy = tokenizer(
        ["自动喷水灭火系统的设计工作压力是多少", "防火卷帘应下降到什么位置"],
        ["系统设计工作压力不应小于 0.6 MPa。", "防火卷帘下降至距地面 1.8 米处。"],
        padding=True,
        truncation="only_second",
        max_length=32,
        return_tensors="pt",
    )
    inputs = {"input_ids": dummy["input_ids"], "attention_mask": dummy["attention_mask"]}

    fp32_path = out_dir / "model.onnx"
    started = time.perf_counter()
    # 用 TorchScript 导出器（dynamo=False）：dynamo 路径在这类 xlm-roberta 分类头上会留下
    # 一个静态 batch 形状，量化的 shape-inference 一跑就报 "dimension 0: (1024) vs (1)"。
    torch.onnx.export(
        model,
        tuple(inputs.values()),
        str(fp32_path),
        input_names=list(inputs.keys()),
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"FP32 ONNX 导出完成：{fp32_path}（{time.perf_counter() - started:.0f}s，"
          f"{fp32_path.stat().st_size / 1024 / 1024:.0f} MB）")

    if args.no_quantize:
        return 0

    from onnxruntime.quantization import QuantType, quantize_dynamic

    int8_path = out_dir / "model.int8.onnx"
    started = time.perf_counter()
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
        # per-channel 量化对 transformer 的线性层更友好，精度掉得少
        per_channel=True,
        reduce_range=False,
    )
    print(f"int8 量化完成：{int8_path}（{time.perf_counter() - started:.0f}s，"
          f"{int8_path.stat().st_size / 1024 / 1024:.0f} MB）")

    # 用 int8 版覆盖 model.onnx：reranker 只认这个文件名
    shutil.move(str(int8_path), str(fp32_path))

    # 验一遍：同一批样本上，int8 与 FP32 的排序必须一致，分数相关性要够高
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    feeds = {name: dummy[name].numpy() for name in ("input_ids", "attention_mask")}
    quantized = session.run(None, feeds)[0].reshape(-1)
    with torch.inference_mode():
        reference = model(**dummy).logits.reshape(-1).numpy()

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    corr = float(np.corrcoef(sigmoid(reference), sigmoid(quantized))[0, 1])
    same_order = list(np.argsort(-reference)) == list(np.argsort(-quantized))
    print(f"一致性校验：分数相关系数 {corr:.5f}，排序一致={same_order}")
    print(f"\n让服务走 ONNX：在 .env 加 RERANK_ONNX_PATH={out_dir}")
    return 0 if corr > 0.99 else 1


if __name__ == "__main__":
    raise SystemExit(main())
