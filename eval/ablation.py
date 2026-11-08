#!/usr/bin/env python3
"""消融矩阵：六档检索 × 50 题，一次跑完，出表。

为什么要单独一个脚本
------------------
`eval.py` 回答「现在这套配置行不行」，它打 `/query`、花 token、还会被 MIN_SCORE 影响。
消融要回答的是另一个问题：「每一个新增部件各自值多少个点、各自要多少毫秒」。
所以这里只打 `/search`（不调大模型、不过阈值）、固定同一批题、同一份语料，
把档位当成唯一的自变量。

为什么每档都要重跑整套题，而不是在同一批结果上重排
------------------------------------------------
那样会把「重排」和「重新召回」混在一起。可信的对照是：**换档 → 重新检索 → 记指标**。
代价是每档 50 次请求；收益是这个矩阵能直接放进博客和面试，不怕被追问「你的对照怎么做的」。

用法
----
    uvicorn app.main:app --port 8031          # 先起服务（配置用真实语料）
    python -m eval.ablation --base-url http://127.0.0.1:8031
    python -m eval.ablation --repeat 3        # 延迟多采几轮再取 p95

注意要用 `-m` 模块方式跑：直接 `python eval/ablation.py` 会把 `eval/` 加进 sys.path 首位，
于是同目录的 `eval.py` 被当成名为 eval 的模块，`from eval.eval import ...` 直接报错。

产物
----
    eval/results/ablation-<YYYYMMDD-HHMM>.json   每档每题的排名、分数、分段耗时
    eval/results/ablation-<YYYYMMDD-HHMM>.md     可直接贴进博客的表格与结论
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if __package__ in (None, ""):  # 直接跑脚本时的友好提示，见模块 docstring
    raise SystemExit("请用模块方式运行：python -m eval.ablation --base-url http://127.0.0.1:8031")

from app.config import get_settings  # noqa: E402
from eval.eval import Case, load_cases, load_chunks, resolve_gold  # noqa: E402

# 顺序就是叙事顺序：基线 → 加 BM25 → 换成 RRF → 单独加重排 → 两件事一起上。
# `bm25`（单路）不在题面要求的五档里，但它能回答「BM25 能不能单独扛」，
# 默认不跑，`--with-extra` 打开。
LADDER: list[tuple[str, str, str]] = [
    ("vector", "纯向量", "基线：bge-m3 余弦 top20"),
    ("hybrid_sum", "+BM25", "向量 + BM25，两路分数归一化后加权（0.5/0.5）"),
    ("hybrid_rrf", "+RRF", "向量 + BM25，RRF(k=60) 按名次融合（当前默认）"),
    ("vector_rerank", "+rerank", "向量 top20 → bge-reranker-v2-m3 重排"),
    ("hybrid_rrf_rerank", "+RRF+rerank", "RRF top20 → 重排（质量上限档，非默认）"),
    ("bm25", "纯 BM25", "对照：只用稀疏检索（诊断用，不作候选方案）"),
]

EXTRA_MODES = ("bm25",)
# 会走重排的档位：报告里「重排覆盖」一列只对它们有意义
RERANK_MODES = frozenset({"vector_rerank", "hybrid_rrf_rerank"})


@dataclass
class Observation:
    case_id: str
    mode: str
    rank_loose: int | None = None  # 宽松：任一承载切片的名次
    rank_strict: int | None = None  # 严格：仅主承载切片
    top_score: float = 0.0
    latencies: list[float] = field(default_factory=list)
    rerank_latencies: list[float] = field(default_factory=list)
    degraded: list[str] = field(default_factory=list)
    channels: dict[str, int] = field(default_factory=dict)  # 候选里各通道的命中数
    rerank_candidates: int = 0  # 实际送进重排的条数（0 = 被预算闸门跳过或本档不重排）


def http_search(
    base_url: str,
    question: str,
    top_k: int,
    mode: str,
    api_key: str | None,
    timeout: int,
    rerank_budget_ms: int | None = None,
) -> dict:
    body: dict = {"question": question, "top_k": top_k, "mode": mode}
    if rerank_budget_ms is not None:
        body["rerank_budget_ms"] = rerank_budget_ms
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    request = urllib.request.Request(
        f"{base_url}/api/v1/search", data=payload, headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise SystemExit(f"/search 返回 {exc.code}（mode={mode}）：{detail}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"连不上 {base_url}（{exc.reason}）。先启动服务：uvicorn app.main:app --port 8031"
        ) from exc


def http_server_status(base_url: str, api_key: str | None, timeout: int) -> dict:
    """取服务端实际的检索配置。

    报告的抬头必须描述**被评测的那个服务**：本地 env 里的 RERANK_MAX_LENGTH 跟服务端
    完全可能不是一回事（扫窗口时就是各自进程一个值），拿本地值写报告等于写错参数。
    """
    headers = {"X-API-Key": api_key} if api_key else {}
    request = urllib.request.Request(f"{base_url}/api/v1/retrieval", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 - 老版本服务没这个端点，不该让评测跑不起来
        return {}


def prepare_gold(cases: list[Case], rows: list[tuple[dict, str]]) -> dict[str, tuple[list, list]]:
    """锚点 → 切片 key。宽松/严格两套口径与 eval.py 一致，否则数不可比。"""
    table: dict[str, tuple[list, list]] = {}
    for case in cases:
        loose: list[tuple[str, int]] = []
        strict: list[tuple[str, int]] = []
        for gold in case.golds:
            keys, level = resolve_gold(rows, gold)
            if not keys:
                print(f"  ! {case.id} 锚点未解析：{gold.anchor[:30]}…", file=sys.stderr)
                continue
            if level > 0:
                print(f"  ~ {case.id} 锚点降级(level={level})", file=sys.stderr)
            for key in keys:
                if key not in loose:
                    loose.append(key)
                if not gold.alternate and key not in strict:
                    strict.append(key)
        table[case.id] = (loose, strict)
    return table


def rank_of(hits: list[dict], keys: list[tuple[str, int]]) -> int | None:
    positions = [
        index
        for index, hit in enumerate(hits, start=1)
        if (str(hit.get("doc_id") or ""), int(hit.get("chunk_index") or 0)) in keys
    ]
    return min(positions) if positions else None


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    # 最近秩法：50 个样本算 p95 取第 48 个，比线性插值保守，也更接近线上体感
    index = min(len(ordered) - 1, max(0, round(q * len(ordered)) - 1))
    return ordered[index]


def summarize(
    observations: list[Observation],
    gold_table: dict[str, tuple[list, list]],
    k: int,
    depth: int,
) -> dict:
    # 分母是「gold 能反查到的有答案题」，不是「命中的题」——后者会让 hit 永远偏高
    resolved = [item for item in observations if gold_table.get(item.case_id, ([], []))[0]]
    strict_resolved = [item for item in observations if gold_table.get(item.case_id, ([], []))[1]]

    def hit_rate(items: list[Observation], name: str, cutoff: int) -> float:
        if not items:
            return 0.0
        return round(
            sum(1 for item in items if (getattr(item, name) or 10**6) <= cutoff) / len(items), 4
        )

    def mrr(items: list[Observation], name: str) -> float:
        if not items:
            return 0.0
        return round(
            sum(1.0 / getattr(item, name) for item in items if getattr(item, name)) / len(items), 4
        )

    latencies = [value for item in observations for value in item.latencies]
    rerank_ms = [value for item in observations for value in item.rerank_latencies]
    channels: dict[str, int] = {}
    for item in observations:
        for name, count in item.channels.items():
            channels[name] = channels.get(name, 0) + count

    return {
        "questions": len(resolved),
        f"hit@{k}": hit_rate(resolved, "rank_loose", k),
        f"hit@{k}_strict": hit_rate(strict_resolved, "rank_strict", k),
        f"hit@{depth}": hit_rate(resolved, "rank_loose", depth),
        "mrr": mrr(resolved, "rank_loose"),
        "mrr_strict": mrr(strict_resolved, "rank_strict"),
        "latency_p50_ms": round(percentile(latencies, 0.50), 1),
        "latency_p95_ms": round(percentile(latencies, 0.95), 1),
        "samples": len(latencies),
        "rerank_p50_ms": round(percentile(rerank_ms, 0.50), 1) if rerank_ms else None,
        "rerank_p95_ms": round(percentile(rerank_ms, 0.95), 1) if rerank_ms else None,
        "channels": channels,
        "degraded": sum(1 for item in observations if item.degraded),
        # 重排覆盖：真有重排发生的问题占比。延迟降了但覆盖率也降了，等于没优化，
        # 所以这两列必须并排看。
        "rerank_coverage": round(
            sum(1 for item in observations if item.rerank_candidates) / len(observations), 4
        )
        if observations
        else 0.0,
        "rerank_candidates_avg": round(
            sum(item.rerank_candidates for item in observations) / len(observations), 1
        )
        if observations
        else 0.0,
    }


def collect(
    base_url: str,
    cases: list[Case],
    gold_table: dict[str, tuple[list, list]],
    modes: list[str],
    depth: int,
    repeat: int,
    api_key: str | None,
    timeout: int,
    rerank_budget_ms: int | None = None,
) -> dict[str, list[Observation]]:
    results: dict[str, list[Observation]] = {}

    print(f"预热：{len(modes)} 档各打一次（把模型加载时间挡在计时之外）…", file=sys.stderr)
    for mode in modes:
        http_search(
            base_url, cases[0].question, depth, mode, api_key, timeout, rerank_budget_ms
        )

    for mode in modes:
        results[mode] = [
            Observation(case_id=case.id, mode=mode)
            for case in cases
            if case.type == "answerable"
        ]
        by_id = {item.case_id: item for item in results[mode]}

        for round_index in range(repeat):
            print(f"\n[{mode}] 第 {round_index + 1}/{repeat} 轮", file=sys.stderr)
            for index, case in enumerate(cases, start=1):
                if case.type != "answerable":
                    continue  # 无答案题在检索层没有 rank，拒答指标交给 eval.py
                loose, strict = gold_table.get(case.id, ([], []))
                body = http_search(
                    base_url, case.question, depth, mode, api_key, timeout, rerank_budget_ms
                )
                hits = body.get("hits") or []
                item = by_id[case.id]
                item.latencies.append(float(body.get("elapsed_ms") or 0.0))
                rerank_ms = (body.get("timings") or {}).get("rerank_ms")
                if rerank_ms:
                    item.rerank_latencies.append(float(rerank_ms))
                if body.get("degraded"):
                    item.degraded = body["degraded"]
                item.rerank_candidates = max(
                    item.rerank_candidates, int(body.get("rerank_candidates") or 0)
                )
                if round_index == 0:  # 排名只记首轮，后续轮次纯采样延迟
                    item.rank_loose = rank_of(hits, loose)
                    item.rank_strict = rank_of(hits, strict)
                    item.top_score = float(hits[0].get("score") or 0.0) if hits else 0.0
                    for hit in hits:
                        for channel in hit.get("channels") or []:
                            item.channels[channel] = item.channels.get(channel, 0) + 1
                mark = "✓" if rank_of(hits, loose) else "✗"
                print(
                    f"  [{index}/{len(cases)}] {case.id} {mark} "
                    f"{item.latencies[-1]:.0f}ms",
                    file=sys.stderr,
                )
    return results


def build_markdown(
    summary: dict[str, dict],
    results: dict[str, list[Observation]],
    cases: list[Case],
    settings,
    k: int,
    depth: int,
    stamp: str,
    corpus: tuple[int, int],
    budget_ms: int | None = None,
    server: dict | None = None,
    top_n: int | None = None,
) -> str:
    questions = [case for case in cases if case.type == "answerable"]
    server = server or {}
    top_n = top_n or settings.rerank_top_n
    lines: list[str] = [
        f"# 检索消融矩阵 · {stamp}",
        "",
        f"- 语料：{corpus[0]} 篇 / {corpus[1]} 切片；题库：{len(questions)} 道有答案题"
        "（无答案题不参与检索排名，见 `eval/results/*.json` 的拒答指标）",
        (
            f"- Embedding：`{settings.embedding_model}`；重排："
            f"`{Path(str(server.get('rerank', {}).get('model') or settings.rerank_model)).name}`"
            f"（backend={server.get('rerank', {}).get('backend', '?')}, "
            f"max_length={server.get('rerank', {}).get('max_length', '?')}, "
            f"top_n={top_n}, batch={settings.rerank_batch_size}）"
        ),
        f"- 每档召回深度 {depth}，hit@{k} / MRR 按「第一个承载切片的名次」算；"
        "严格口径剔除 `alternate` 承载文档",
        "- 延迟是服务端 `/search` 的 `elapsed_ms`（含 Embedding 与重排，不含网络与生成）",
        (
            f"- **重排闸门：{budget_ms} ms**。预算换算不出足够候选时整档跳过，"
            "所以这一轮看的不是「重排最好能多准」，而是「预算内还能留下多少准」"
            if budget_ms
            else "- 重排闸门：**关闭**（这一轮是质量上限，不看延迟预算）"
        ),
        "",
        "## 主表",
        "",
        f"| 档位 | 说明 | hit@{k}（严） | hit@{k}（宽） | MRR | hit@{depth} | 重排覆盖 | p50 | p95 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, name, note in LADDER:
        if mode not in summary:
            continue
        row = summary[mode]
        covered = (
            f"{row['rerank_coverage']:.0%} / {row['rerank_candidates_avg']:.1f} 条"
            if mode in RERANK_MODES
            else "—"
        )
        lines.append(
            f"| `{name}` | {note} | {row[f'hit@{k}_strict']:.1%} | {row[f'hit@{k}']:.1%} | "
            f"{row['mrr']:.4f} | {row[f'hit@{depth}']:.1%} | {covered} | "
            f"{row['latency_p50_ms']:.0f} ms | {row['latency_p95_ms']:.0f} ms |"
        )

    lines += [
        "",
        "## 分阶段增量",
        "",
        f"| 对比 | Δhit@{k}（宽） | ΔMRR | Δp95 |",
        "| --- | ---: | ---: | ---: |",
    ]
    pairs = [
        ("+BM25（相对基线）", "hybrid_sum", "vector"),
        ("+RRF（相对 +BM25）", "hybrid_rrf", "hybrid_sum"),
        ("+RRF（相对基线）", "hybrid_rrf", "vector"),
        ("+rerank（相对基线）", "vector_rerank", "vector"),
        ("+RRF+rerank（相对基线）", "hybrid_rrf_rerank", "vector"),
        ("+RRF+rerank（相对 +RRF）", "hybrid_rrf_rerank", "hybrid_rrf"),
    ]
    for label, left, right in pairs:
        if left not in summary or right not in summary:
            continue
        a, b = summary[left], summary[right]
        lines.append(
            f"| {label} | {a[f'hit@{k}'] - b[f'hit@{k}']:+.1%} | {a['mrr'] - b['mrr']:+.4f} | "
            f"{a['latency_p95_ms'] - b['latency_p95_ms']:+.0f} ms |"
        )

    # 翻盘明细：把「哪几道题因为这一步上车/下车」列出来，比任何百分比都好讲
    lines += ["", "## 谁上车、谁下车（相对基线）", ""]
    base_ranks = {item.case_id: item.rank_loose for item in results.get("vector", [])}
    if "vector" not in results:
        lines.append(
            "_本轮没有跑 `vector` 基线，无法计算翻盘明细。要看这一段，"
            "请用 `--with-extra` 跑完整矩阵。_"
        )
    for mode, name, _note in [] if "vector" not in results else LADDER[1:]:
        if mode not in results:
            continue
        ranks = {item.case_id: item.rank_loose for item in results[mode]}
        gained = [qid for qid in ranks if ranks[qid] and not base_ranks.get(qid)]
        lost = [qid for qid in ranks if base_ranks.get(qid) and not ranks[qid]]
        improved = [
            qid
            for qid in ranks
            if ranks[qid] and base_ranks.get(qid) and ranks[qid] < base_ranks[qid]
        ]
        if not (gained or lost or improved):
            lines.append(f"- **{name}**：与基线排名完全一致，没有题目发生位次变化")
            continue
        detail = []
        if gained:
            detail.append(f"新命中 {len(gained)} 道（{'、'.join(gained[:8])}）")
        if lost:
            detail.append(f"丢失 {len(lost)} 道（{'、'.join(lost[:8])}）")
        if improved:
            detail.append(f"位次前移 {len(improved)} 道")
        lines.append(f"- **{name}**：" + "；".join(detail))

    lines += [
        "",
        "## 怎么看这张表",
        "",
        "- **hit@k 是产品指标**：进不了前 k，模型就看不到原文。它和 hit@20 的差距说明「排序」还有得治，"
        "而不是「召回不够」——这两件事的修法完全不同。",
        "- **MRR 比 hit 敏感**：把答案从第 4 名提到第 1 名，hit 不动、MRR 会动。只看 hit 会漏掉这类改进。",
        "- **延迟要成对看**：涨的点数如果只买来 p95 从 150ms 到 2s，在交互式问答里是亏的。"
        "预算线应该由产品定，不是由别人的博客定。",
        "- **重排覆盖是这次新加的列**：交叉编码器的成本是「候选数 × 每候选 token 数」，"
        "预算不够时它会被整档跳过。覆盖率一栏如果很低，`+rerank` 那行的分数其实来自融合顺序——"
        "把它当「重排的效果」写进博客，就是拿没跑的东西邀功。",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="检索消融矩阵")
    parser.add_argument("--base-url", default="http://127.0.0.1:8031")
    parser.add_argument("--questions", default=str(Path(__file__).parent / "questions.yaml"))
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument(
        "--depth", type=int, default=20, help="每档召回深度，应与服务端 recall_candidates 对齐"
    )
    parser.add_argument("--repeat", type=int, default=1, help="每档重复轮数（排名只取首轮）")
    parser.add_argument("--modes", default="", help="只跑指定档位，逗号分隔")
    parser.add_argument("--with-extra", action="store_true", help="额外跑纯 BM25 档（诊断对照）")
    parser.add_argument("--chroma-dir", default=None)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--rerank-budget",
        type=int,
        default=None,
        help="重排时间闸门（毫秒）。不给 = 用服务端配置；给 0 = 不设限（质量上限）",
    )
    parser.add_argument("--out-dir", default=str(Path(__file__).parent / "results"))
    args = parser.parse_args()

    settings = get_settings()
    cases = load_cases(Path(args.questions))
    answerable = [case.id for case in cases if case.type == "answerable"]
    if not answerable:
        raise SystemExit("题库里没有 answerable 题，消融没有分母")

    modes = [mode for mode, _name, _note in LADDER if args.with_extra or mode not in EXTRA_MODES]
    if args.modes:
        modes = [item.strip() for item in args.modes.split(",") if item.strip()]

    rows = load_chunks(args.chroma_dir or settings.chroma_dir, settings.collection_name)
    documents = len({str(meta.get("doc_id") or "") for meta, _ in rows})
    print(f"库：{documents} 篇 / {len(rows)} 切片 | 题：{len(answerable)} 道 | 档位：{', '.join(modes)}")
    gold_table = prepare_gold(cases, rows)
    server = http_server_status(args.base_url, settings.api_key, args.timeout)
    if server:
        print(f"服务端：mode={server.get('mode')} 重排={server.get('rerank')}")
    else:
        print("  ! 读不到服务端 /retrieval，报告里的重排参数将回落到本地 env（可能不一致）")

    started = time.perf_counter()
    results = collect(
        args.base_url,
        cases,
        gold_table,
        modes,
        args.depth,
        args.repeat,
        settings.api_key,
        args.timeout,
        args.rerank_budget,
    )
    summary = {
        mode: summarize(results[mode], gold_table, args.k, args.depth) for mode in modes
    }

    print("")
    header = (
        f"{'档位':<16}{f'hit@{args.k}(宽)':>12}{f'hit@{args.k}(严)':>12}"
        f"{'MRR':>9}{'p50':>9}{'p95':>9}"
    )
    print(header)
    print("-" * len(header))
    for mode, name, _note in LADDER:
        if mode not in summary:
            continue
        row = summary[mode]
        print(
            f"{name:<16}{row[f'hit@{args.k}']:>12.1%}{row[f'hit@{args.k}_strict']:>12.1%}"
            f"{row['mrr']:>9.4f}{row['latency_p50_ms']:>8.0f}ms{row['latency_p95_ms']:>8.0f}ms"
        )
    print(f"\n总耗时 {time.perf_counter() - started:.0f}s")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "base_url": args.base_url,
        "k": args.k,
        "depth": args.depth,
        "repeat": args.repeat,
        "config": {
            "embedding_model": settings.embedding_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "rrf_k": settings.rrf_k,
            "bm25_weight": settings.bm25_weight,
            "rerank_model": settings.rerank_model,
            "rerank_max_length": settings.rerank_max_length,
            "rerank_batch_size": settings.rerank_batch_size,
            "rerank_onnx": settings.rerank_onnx_path,
            "rerank_top_n": server.get("rerank_top_n") or settings.rerank_top_n,
            "rerank_budget_ms": args.rerank_budget,
        },
        "corpus": {"documents": documents, "chunks": len(rows)},
        "summary": summary,
        "ranks": {
            mode: {item.case_id: item.rank_loose for item in results[mode]} for mode in modes
        },
    }
    json_path = out_dir / f"ablation-{stamp}.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md_path = out_dir / f"ablation-{stamp}.md"
    md_path.write_text(
        build_markdown(
            summary,
            results,
            cases,
            settings,
            args.k,
            args.depth,
            stamp,
            (documents, len(rows)),
            args.rerank_budget,
            server,
            server.get("rerank_top_n") or settings.rerank_top_n,
        ),
        encoding="utf-8",
    )
    print(f"\n结果：{json_path}")
    print(f"报告：{md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
