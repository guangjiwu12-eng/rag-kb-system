#!/usr/bin/env python3
"""重排延迟压测：把 max_length / batch / 后端这三个旋钮扫一遍，出预算表。

为什么单独压一段
--------------
整条链路的 p95 里，向量检索（Embedding 一次前向 + HNSW）在本机是百毫秒量级，
重排是几十倍——所以「能不能压进预算」实际上只取决于重排这一段。把它单独拎出来扫参数，
才能说清「多花的 300ms 花在哪、能换回多少」。

候选 token 总量才是成本
--------------------
cross-encoder 的开销 ≈ 候选数 × 每个候选的 token 数 × 模型规模。候选数（20）和模型
（v2-m3）都是题面定死的，剩下唯一能动的就是 `max_length`。所以这张表的核心结论通常是
「把 800 字的切片截到 256 token，砍掉的主要是切片尾部」——到底值不值，看消融矩阵。

用法
----
    python scripts/bench_rerank.py                       # 默认扫 128/256/512 × batch 4/8/16
    python scripts/bench_rerank.py --lengths 192 --iters 5
    python scripts/bench_rerank.py --onnx models/bge-reranker-v2-m3-onnx-int8
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.reranker import CrossEncoderReranker  # noqa: E402


def sample_passages(count: int, chars: int) -> list[str]:
    """从真实语料里取切片，长度按 chars 截断。

    用真切片而不是随机字符串：中文 token/字的比例和标点密度对 token 数影响很大，
    假的样本会把延迟估得漂亮很多。
    """
    from app.config import get_settings as _s
    from eval.eval import load_chunks

    settings = _s()
    try:
        rows = load_chunks(settings.chroma_dir, settings.collection_name)
    except SystemExit:
        rows = []
    texts = [text for _meta, text in rows if len(text) >= chars][:count]
    if len(texts) < count:  # 库是空的也要能跑（用同一段话复制）
        fallback = (
            "自动喷水灭火系统的设计工作压力应根据系统类型确定，且不应小于 0.6 MPa。"
            "条文说明指出，压力不足会导致喷头开启后无法形成有效水雾，进而失去控火能力。"
        )
        texts = (texts + [fallback * 4] * count)[:count]
    return [text[:chars] for text in texts]


def bench(
    label: str,
    reranker: CrossEncoderReranker,
    passages: list[str],
    iters: int,
    query: str = "自动喷水灭火系统的设计工作压力不应小于多少",
) -> dict:
    reranker.rerank(query, passages)  # 预热：第一次要分配内存、编译图
    samples = []
    for _ in range(iters):
        started = time.perf_counter()
        reranker.rerank(query, passages)
        samples.append((time.perf_counter() - started) * 1000)
    ordered = sorted(samples)
    p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95) - 1)]
    return {
        "label": label,
        "mean_ms": round(statistics.fmean(samples), 1),
        "p50_ms": round(statistics.median(samples), 1),
        "p95_ms": round(p95, 1),
        "min_ms": round(ordered[0], 1),
        "max_ms": round(ordered[-1], 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="重排延迟压测")
    parser.add_argument("--candidates", type=int, default=20, help="每次重排多少条候选")
    parser.add_argument(
        "--candidate-counts",
        default="",
        help="另外扫一遍候选数（逗号分隔，例如 4,8,12,20）：这是除 max_length 外最强的旋钮",
    )
    parser.add_argument("--chars", type=int, default=800, help="候选切片截断到多少字符")
    parser.add_argument("--lengths", default="128,256,512", help="重排 max_length 扫描点")
    parser.add_argument("--batches", default="4,8,16", help="批大小扫描点")
    parser.add_argument("--iters", type=int, default=3, help="每档重复次数（取 p95 用）")
    parser.add_argument("--threads", type=int, default=None, help="torch 线程数")
    parser.add_argument("--onnx", default=None, help="int8 ONNX 目录（给上则只测 ONNX）")
    parser.add_argument("--budget", type=float, default=500.0, help="总预算（毫秒），用于判定")
    args = parser.parse_args()

    settings = get_settings()
    lengths = [int(item) for item in args.lengths.split(",") if item.strip()]
    batches = [int(item) for item in args.batches.split(",") if item.strip()]
    counts = [int(item) for item in args.candidate_counts.split(",") if item.strip()]
    pool = max([args.candidates, *counts])
    passages = sample_passages(pool, args.chars)
    print(
        f"候选 {len(passages)} 条 × 截断 {args.chars} 字；"
        f"线程 {args.threads or '默认'}；模型 {Path(settings.rerank_model).name}",
        flush=True,
    )

    rows: list[dict] = []
    if args.onnx:
        reranker = CrossEncoderReranker(
            settings.rerank_model,
            onnx_path=args.onnx,
            batch_size=batches[0],
            max_length=lengths[0],
            num_threads=args.threads,
        )
        reranker.load()
        for length in lengths:
            for batch in batches:
                reranker.max_length = length
                reranker.batch_size = batch
                rows.append(
                    bench(
                        f"onnx-int8 L{length} B{batch} N{len(passages)}",
                        reranker,
                        passages,
                        args.iters,
                    )
                )
                print(f"  {rows[-1]}", flush=True)
    else:
        reranker = CrossEncoderReranker(
            settings.rerank_model,
            batch_size=batches[0],
            max_length=lengths[0],
            num_threads=args.threads,
        )
        reranker.load()
        print(f"加载耗时 {reranker.load_seconds:.1f}s，后端 {reranker.backend}", flush=True)
        for length in lengths:
            for batch in batches:
                reranker.max_length = length
                reranker.batch_size = batch
                rows.append(
                    bench(
                        f"torch L{length} B{batch} N{len(passages)}",
                        reranker,
                        passages,
                        args.iters,
                    )
                )
                print(f"  {rows[-1]}", flush=True)

    # 第二张表：固定 max_length，只改候选数。用来回答「预算只够排 6 条，够不够」——
    # 消融矩阵那边再补上「排 6 条还剩下多少 hit@4」。
    if counts:
        tag = "onnx-int8" if args.onnx else "torch"
        for length in lengths:
            for count in counts:
                reranker.max_length = length
                reranker.batch_size = max(1, min(8, count))
                rows.append(
                    bench(
                        f"{tag} L{length} N{count}",
                        reranker,
                        passages[:count],
                        args.iters,
                    )
                )
                print(f"  {rows[-1]}", flush=True)

    print("")
    print(f"{'配置':<22}{'mean':>9}{'p50':>9}{'p95':>9}{'能否 < ' + str(int(args.budget)) + 'ms':>14}")
    print("-" * 63)
    for row in rows:
        verdict = "✓" if row["p95_ms"] < args.budget else "✗"
        print(
            f"{row['label']:<22}{row['mean_ms']:>8.0f}ms{row['p50_ms']:>8.0f}ms"
            f"{row['p95_ms']:>8.0f}ms{verdict:>14}"
        )
    best = min(rows, key=lambda item: item["p95_ms"])
    print(f"\n最快配置：{best['label']} p95={best['p95_ms']:.0f}ms")
    print("注意：这里是重排单段耗时；端到端还要加上向量检索（本机约 150ms）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
