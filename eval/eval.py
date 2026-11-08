#!/usr/bin/env python3
"""RAG 评测：检索层打 /api/v1/search，问答层打 /api/v1/query。

为什么必须分成两条通道
--------------------
- `/query` 返回的 citations 已经过 MIN_SCORE 过滤。被阈值拦掉的命中在它眼里等于
  「没检索到」，用它算 recall 会把「检索到了但分数低」和「压根没检索到」混成一件事，
  rank 随之失真。
- `/search` 不过阈值、不调模型，返回原始 top-K 排名，才是测召回的通道；
  `/query` 的 `abstained` 才是测拒答行为的通道。

两条通道合起来，才能同时回答「检索能力如何」和「拒答策略是否合理」。

三套口径
--------
同一批命中，三种算法，一次采集：

1. 严格：只认题目标注的**主承载文档**（gold 里没写 `alternate: true` 的那些锚点）。
   这是最保守的数字，和「一道题只许有一个答案文件」的老口径等价。
2. 宽松：任一**承载文档**都算。当同一句条文在两部标准里都写了（GB50974 与 GB50261
   的「压力表量程 2 倍～2.5 倍」就是逐字相同的两条），只认其中一个文件等于把
   「正确答案存在多处」判成检索失败。
3. 端到端：检索命中（宽松口径）**且**模型确实作答。这个数字才能回答
   「用户问一句话，到底有多少道题真的拿到了答案」。

为什么不直接只报宽松
--------------------
宽松会变好看，所以必须并列严格。两者的差额就是「标注口径能解释多少」，那是分析结论，
不是成绩。只报其中一个，等于用口径偷数字。

第三种失败模式
--------------
`answerable_refused_rate`：40 道有答案题里，模型拒答的比例。
再拆成两支——
- `refused_after_miss_rate`：检索没排进 top-k，模型看不到原文 → 拒答。这类既不进
  误拒率（那是阈值口径）也不进幻觉率，是老表里没有位置的失败模式。
- `refused_despite_hit_rate`：原文就在 top-k 里，模型还是拒答 → Prompt 或长上下文问题。

用法
----
    uvicorn app.main:app            # 先起服务
    python eval/eval.py             # 双路评测 + 出报告
    python eval/eval.py --sweep     # 扫 MIN_SCORE，挑 F1 最高的阈值
    python eval/eval.py --no-query  # 只测检索层（不花 token）
    python eval/eval.py --set-baseline   # 把本轮结果定为 baseline

产物
----
    eval/results/<YYYYMMDD>-<hash>.json    指标与配置快照（hash 含 MIN_SCORE/top_k）
    eval/results/latest.json               本轮结果的固定别名，供 diff_eval.py 用
    eval/results/cases.csv                 每题一行：名次、口径归类、是否拒答
    eval/results/failures.md               未命中清单 + 归因栏（沿用上一轮手写的）
    eval/results/citation_review.md        引用抽检样本（首次生成，之后不覆盖）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from app.config import get_settings  # noqa: E402

_WS = re.compile(r"\s+")
MIN_ANCHOR_CHARS = 8

# 评测只打本机服务，一律不走代理。有些环境里 http_proxy 指向一个本地代理端口，
# urllib 会把 127.0.0.1 的请求也塞进去，结果是 502 —— 看起来像服务挂了，其实是代理回的。
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

# 模型自主拒答的启发式特征。注意这只是启发式：真正的判定要人看，误判成本远低于
# 把「模型兜住了」误记成「拒答失败」——后者会让你去调一个本来没问题的阈值。
ABSTAIN_HINTS = (
    "无法回答",
    "无法确定",
    "未提及",
    "没有提及",
    "未涉及",
    "资料中未",
    "缺少相关",
    "没有相关",
    "不足以回答",
)


def looks_abstained(answer: str | None) -> bool:
    return any(hint in (answer or "") for hint in ABSTAIN_HINTS)


# --------------------------------------------------------------------- 数据结构
@dataclass
class Gold:
    """一条承载锚点。

    `alternate=True` 表示「同一事实的另一个合法承载文档」：它在宽松口径里算命中，
    在严格口径里不算。加这个标记之前必须回库核对原文——**同主题**和**同一条规定**
    是两件事，把前者标成后者就是放宽口径。
    """

    anchor: str
    doc: str | None = None
    page: int | None = None
    # 按条切片之后，一条长条文会跨两页，锚点落在哪一页取决于切片器，不取决于标注人。
    # 给个区间就不用每次为了「到底该写 88 还是 89」重标一遍。
    page_end: int | None = None
    alternate: bool = False

    @property
    def pages(self) -> set[int] | None:
        if self.page is None:
            return None
        end = self.page_end if self.page_end is not None else self.page
        low, high = sorted((self.page, end))
        return set(range(low, high + 1))


@dataclass
class Case:
    id: str
    question: str
    type: str = "answerable"
    answer: str | None = None
    golds: list[Gold] = field(default_factory=list)
    note: str | None = None


@dataclass
class Row:
    case: Case
    hits: list[dict] = field(default_factory=list)  # /search 原始命中
    gold_keys: list[tuple[str, int]] = field(default_factory=list)  # 宽松：全部承载切片
    strict_keys: list[tuple[str, int]] = field(default_factory=list)  # 严格：仅主承载
    unresolved: list[Gold] = field(default_factory=list)
    rank: int | None = None  # 宽松口径下第一个承载切片在 hits 里的位次（1 基）
    rank_strict: int | None = None
    top_score: float = 0.0
    abstained: bool | None = None  # /query 的真实拒答行为；None 表示没打
    llm_abstained: bool | None = None  # 未被阈值拦下时，模型自己是否说"不知道"
    answer: str | None = None
    citations: list[dict] = field(default_factory=list)
    latency_ms: float = 0.0

    @property
    def is_answerable(self) -> bool:
        return self.case.type == "answerable"


# --------------------------------------------------------------------- 工具
def norm(text: str) -> str:
    """去空白后比对：PDF 抽取的换行、缩进不该影响锚点匹配。"""
    return _WS.sub("", text or "")


def http_json(url: str, payload: dict, api_key: str | None, timeout: int = 120) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with _OPENER.open(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise SystemExit(f"{url} 返回 {exc.code}：{detail}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"连不上 {url}（{exc.reason}）。先启动服务：uvicorn app.main:app"
        ) from exc


def http_get_json(url: str, api_key: str | None, timeout: int = 15) -> dict:
    headers = {"X-API-Key": api_key} if api_key else {}
    request = urllib.request.Request(url, headers=headers, method="GET")
    with _OPENER.open(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def check_server(base_url: str, api_key: str | None, settings) -> dict:
    """开跑前先核对服务端身份。

    这里拦两件会被日志漏掉的事：
    1. 打到了别的实例（端口上残留的旧服务）——两边模式不同，指标却看起来正常。
    2. 请求了 rerank 档位，但重排模型没加载成功。服务端只打一条 WARNING 然后降级为融合顺序，
       `/search` 照常返回结果——于是 `+rerank` 档位实际等于 `+RRF` 档位，消融矩阵直接失真。
       这种事必须在这里硬失败，而不是等博客写完之后被面试官问出来。
    """
    try:
        health = http_get_json(f"{base_url}/api/v1/health", api_key)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"连不上 {base_url}/api/v1/health（{exc}）。先启动服务：uvicorn app.main:app"
        ) from exc

    wanted = getattr(settings, "retrieval_mode", "vector")
    served = health.get("retrieval_mode") or "vector"
    print(
        f"服务：{health.get('documents')} 篇 / {health.get('chunks')} 片 | 模式={served} "
        f"| rerank_loaded={health.get('rerank_loaded')}"
    )
    if served != wanted:
        raise SystemExit(
            f"服务端检索模式是 {served}，本机配置是 {wanted}——打到了别的实例，"
            "换个端口重起，否则这次评测的编号对不上配置"
        )
    if "rerank" in str(served) and not health.get("rerank_loaded"):
        raise SystemExit(
            f"档位是 {served}，但 rerank_loaded=false。服务端已静默降级为融合顺序，"
            "这轮测出来的不是 rerank 的效果。先确认重排模型权重已就位"
        )
    return health


def config_hash(settings, retrieve_k: int, k: int) -> str:
    """配置指纹。

    必须把 MIN_SCORE 和 top_k 算进来：同库同题只换阈值，指标会整个变一遍，
    而旧版本 hash 不含这两项，换阈值会**覆盖同名 json**——踩过一次，别再踩。
    k 和 retrieve_k 也进指纹，否则 hit@4 和 hit@6 会落到同一个文件上。

    检索模式同样要进：纯向量 / +BM25 / +RRF / +rerank 是六组不同实验，
    只换模式不改其他参数时，前五项的指纹完全相同。不把它算进去，
    消融的六份结果会互相覆盖，最后只剩一份——这是 MIN_SCORE 那个坑的同一形状。
    """
    from app.config import Settings

    mode = getattr(settings, "retrieval_mode", "vector")
    parts = [
        settings.embedding_model,
        str(settings.chunk_size),
        str(settings.chunk_overlap),
        str(retrieve_k),
        str(k),
        f"{settings.min_score:.2f}",
        str(settings.top_k),
        settings.collection_name,
        str(mode),
    ]
    # BM25 / rerank 只在对应模式下才影响结果，但进指纹无害，且能防「同名不同参」
    for extra in ("bm25_k1", "bm25_b", "rrf_k", "rerank_model", "rerank_batch_size"):
        if hasattr(settings, extra):
            parts.append(str(getattr(settings, extra)))
    # Settings 引用一下，避免 linter 认为 import 未使用（也顺带断言配置类存在）
    assert Settings is not None
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:6]


# --------------------------------------------------------------------- 题库
def load_cases(path: Path) -> list[Case]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("缺少 PyYAML：pip install pyyaml") from exc

    if not path.is_file():
        raise SystemExit(f"找不到题库：{path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    if not isinstance(raw, list):
        raise SystemExit("题库格式错误：顶层应为列表")

    cases: list[Case] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise SystemExit(f"第 {index} 题格式错误：应为映射")
        qtype = str(item.get("type") or "answerable")
        if qtype not in ("answerable", "unanswerable"):
            raise SystemExit(f"第 {index} 题 type 只能是 answerable / unanswerable")
        golds = [
            Gold(
                anchor=str(g.get("anchor") or ""),
                doc=g.get("doc") or g.get("file"),
                page=int(g["page"]) if g.get("page") is not None else None,
                page_end=int(g["page_end"]) if g.get("page_end") is not None else None,
                alternate=bool(g.get("alternate") or False),
            )
            for g in (item.get("gold") or [])
        ]
        if qtype == "answerable" and not golds:
            raise SystemExit(f"第 {index} 题（{item.get('id')}）是 answerable，但没写 gold")
        if qtype == "answerable" and all(g.alternate for g in golds):
            raise SystemExit(
                f"第 {index} 题（{item.get('id')}）的 gold 全是 alternate，没有主承载文档——"
                "严格口径会变成空分母，先决定哪一个是主标注"
            )
        if not str(item.get("question") or "").strip():
            raise SystemExit(f"第 {index} 题的 question 为空")
        cases.append(
            Case(
                id=str(item.get("id") or f"q{index:03d}"),
                question=str(item["question"]).strip(),
                type=qtype,
                answer=item.get("answer"),
                golds=golds,
                note=item.get("note"),
            )
        )
    return cases


# --------------------------------------------------------------------- 锚点反查
def load_chunks(chroma_dir: str, collection_name: str) -> list[tuple[dict, str]]:
    """只读全库切片，用于把 anchor 反查成 (doc_id, chunk_index)。不参与指标计算。"""
    import chromadb

    client = chromadb.PersistentClient(path=chroma_dir)
    try:
        collection = client.get_collection(collection_name)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"读不到向量库（{chroma_dir} / {collection_name}）：{exc}") from exc

    data = collection.get(include=["documents", "metadatas"])
    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []
    return [
        (dict(meta), text or "")
        for text, meta in zip(documents, metadatas, strict=False)
        if meta is not None
    ]


def resolve_gold(rows: list[tuple[dict, str]], gold: Gold) -> tuple[list[tuple[str, int]], int]:
    """返回 ([((doc_id, chunk_index), ...)], 层级)。

    一个锚点往往命中**多个**切片：chunk_overlap=150 会让一句话同时落在相邻两片里，
    按页切开的长段也会跨页重复。它们都是合法的「答对所需的原文」，所以全部收进 gold 集合，
    命中其中任意一片即算 retrieval 成功——只认「第一片」会把切片重叠算成检索失败
    （实测会凭空吃掉 20 个点 hit@k）。

    层级用来区分标注质量：0 = 页也对得上，1 = 同文件但页码不对，2 = 落在别的文件里。
    严格/宽松的差别不在层级，而在 gold 上有没有 `alternate` 标记。

    页码两个口径都认：`page` 是 PDF 物理页，`page_printed` 是页脚印的印刷页。规范 PDF
    里两者常常差一个固定偏移（封面、目录没印页码），标注时随手抄哪个都对得上，
    不用为了对齐口径重标一遍；`page_end` 用来兜「一条长条文跨两页」。

    跨标准同义条文（GB50166 6.0.4 ≡ GB51309 7.0.4）在入库时合并成一片，出处记在
    正本切片的 `also_in` 里。命中这种切片同样算答对——文字确实在库里，只是换了件外衣。
    """
    target = norm(gold.anchor)
    if len(target) < MIN_ANCHOR_CHARS:
        raise ValueError(f"锚点太短（{len(target)} 字），至少 {MIN_ANCHOR_CHARS} 字且必须唯一")
    filename = Path(gold.doc).name.casefold() if gold.doc else None
    wanted_pages = gold.pages

    def same_doc(meta: dict) -> bool:
        if filename is None:
            return True
        return Path(str(meta.get("source") or "")).name.casefold() == filename

    def page_ok(meta: dict) -> bool:
        if wanted_pages is None:
            return True
        physical = int(meta.get("page") or 0)
        printed = int(meta.get("page_printed") or 0)
        return physical in wanted_pages or (printed and printed in wanted_pages)

    def alias_ok(meta: dict) -> bool:
        """`also_in` 里记着「这段文字还出现在哪」；对上文件与页码就算同一处证据。"""
        if filename is None or wanted_pages is None:
            return False
        for entry in str(meta.get("also_in") or "").split("；"):
            parts = entry.split(":")
            if len(parts) < 2:
                continue
            if Path(parts[0]).name.casefold() != filename:
                continue
            try:
                if int(parts[1]) in wanted_pages:
                    return True
            except ValueError:
                continue
        return False

    levels: list[tuple[list[tuple[str, int]], int]] = [([], -1) for _ in range(3)]
    for meta, text in rows:
        alias = alias_ok(meta)
        if target not in norm(text) and not alias:
            continue
        key = (str(meta.get("doc_id") or ""), int(meta.get("chunk_index") or 0))
        # 别名命中直接算层级 0：这一片就是那句话（换了个文件名与页码的正本），
        # 再走 same_doc 判断只会因为「文件名不一样」把它降级成 2。
        if alias or (same_doc(meta) and page_ok(meta)):
            levels[0][0].append(key)
        elif same_doc(meta):
            levels[1][0].append(key)
        else:
            levels[2][0].append(key)
    for level, (keys, _unused) in enumerate(levels):
        if keys:
            return keys, level
    return [], -1


def first_rank(hit_keys: list[tuple[str, int]], gold_keys: list[tuple[str, int]]) -> int | None:
    """gold 在检索结果里最早出现在第几位（1 基）；一片都没命中返回 None。

    写成模块级函数而不是循环里的闭包：闭包捕获循环变量，读代码时看不出它绑的是哪一轮。
    """
    ranks = [hit_keys.index(key) + 1 for key in gold_keys if key in hit_keys]
    return min(ranks) if ranks else None


# --------------------------------------------------------------------- 采集
def collect(
    base_url: str,
    cases: list[Case],
    rows: list[tuple[dict, str]],
    retrieve_k: int,
    api_key: str | None,
    with_query: bool,
) -> list[Row]:
    results: list[Row] = []
    for index, case in enumerate(cases, start=1):
        row = Row(case=case)
        for gold in case.golds:
            keys, level = resolve_gold(rows, gold)
            if not keys:
                row.unresolved.append(gold)
                print(f"  ! {case.id} 锚点未解析：{gold.anchor[:30]}…", file=sys.stderr)
                continue
            if level > 0:
                print(f"  ~ {case.id} 锚点降级(level={level})：{gold.anchor[:30]}…", file=sys.stderr)
            bucket = row.gold_keys if not gold.alternate else row.gold_keys
            for key in keys:
                if key not in bucket:
                    bucket.append(key)
                if not gold.alternate and key not in row.strict_keys:
                    row.strict_keys.append(key)

        searched = http_json(
            f"{base_url}/api/v1/search",
            {"question": case.question, "top_k": retrieve_k},
            api_key,
        )
        row.hits = searched.get("hits") or []
        row.latency_ms = float(searched.get("elapsed_ms") or 0)
        row.top_score = max((float(h.get("score") or 0.0) for h in row.hits), default=0.0)

        hit_keys = [(str(h.get("doc_id") or ""), int(h.get("chunk_index") or 0)) for h in row.hits]

        row.rank = first_rank(hit_keys, row.gold_keys)
        row.rank_strict = first_rank(hit_keys, row.strict_keys)

        if with_query:
            answered = http_json(f"{base_url}/api/v1/query", {"question": case.question}, api_key)
            row.abstained = bool(answered.get("abstained"))
            row.answer = answered.get("answer")
            row.citations = answered.get("citations") or []
            # 阈值没拦住时，模型自己可能仍然拒绝作答。这两条路径必须分开记：
            # 混在一起会把「模型兜住了」误判成「拒答能力为 0」。
            row.llm_abstained = False if row.abstained else looks_abstained(row.answer)

        mark = "✓" if row.rank else "✗"
        print(
            f"  [{index}/{len(cases)}] {case.id} {mark} "
            f"rank={row.rank}(严 {row.rank_strict}) top={row.top_score:.3f}",
            file=sys.stderr,
        )
        results.append(row)
    return results


# --------------------------------------------------------------------- 归类
def classify(row: Row, k: int) -> str:
    """按名次给每道有答案题一个机械可复现的归类。

    只归因「检索为什么没中」，不猜模型。人写的归因在 failures.md 里另有一栏，
    两栏对不上就说明人的判断和数字有一边错了。

    排第 7 和排第 14 不是同一类问题：前者加两个名额就能救，后者加名额是自欺欺人
    （k 从 6 加到 14，噪声也一起灌进 Prompt），只能靠 rerank 把对的顶上来。
    """
    if row.case.type != "answerable":
        return "-"
    if row.rank is None:
        return "未进 top20"
    if row.rank > k:
        if row.rank <= k + 3:
            return f"近截断（排第 {row.rank}）"
        return f"排名靠后（第 {row.rank}，需 rerank）"
    if row.rank_strict is None or row.rank_strict > k:
        return f"仅宽松命中（第 {row.rank} 位是替代承载文档）"
    return "命中"


# --------------------------------------------------------------------- 指标
def compute_metrics(rows: list[Row], k: int) -> dict:
    answerable = [r for r in rows if r.is_answerable]
    strict_resolved = [r for r in answerable if r.strict_keys]
    loose_resolved = [r for r in answerable if r.gold_keys]
    unanswerable = [r for r in rows if not r.is_answerable]

    def hit_mrr(pool: list[Row], attr: str) -> tuple[float | None, float | None]:
        if not pool:
            return None, None
        ranks = [getattr(r, attr) for r in pool]
        hits = sum(1 for rank in ranks if rank and rank <= k)
        mrr = sum((1.0 / rank) if (rank and rank <= k) else 0.0 for rank in ranks) / len(pool)
        return round(hits / len(pool), 4), round(mrr, 4)

    strict_hit, strict_mrr = hit_mrr(strict_resolved, "rank_strict")
    loose_hit, loose_mrr = hit_mrr(loose_resolved, "rank")

    metrics: dict[str, float | int | None] = {
        f"strict_hit@{k}": strict_hit,
        "strict_mrr": strict_mrr,
        "strict_n": len(strict_resolved),
        f"loose_hit@{k}": loose_hit,
        "loose_mrr": loose_mrr,
        "loose_n": len(loose_resolved),
    }
    # 差额 = 标注口径能解释的部分。这个数不该被藏起来，它本身就是结论。
    if strict_hit is not None and loose_hit is not None:
        metrics["caliber_gap"] = round(loose_hit - strict_hit, 4)

    scored = [r for r in answerable if r.abstained is not None]
    if scored:
        # 「拒答」必须把两条路径合并：MIN_SCORE=0.3 时阈值路径几乎不触发，
        # 只看 abstained 会得出「0% 拒答」的假象——实际上是模型自己说了"无法回答"。
        refused = [r for r in scored if r.abstained or r.llm_abstained]
        e2e = sum(1 for r in scored if r.rank and r.rank <= k and not (r.abstained or r.llm_abstained))
        metrics["e2e_answer_rate"] = round(e2e / len(scored), 4)
        metrics["e2e_n"] = len(scored)
        metrics["answerable_refused_rate"] = round(len(refused) / len(scored), 4)

        # 拒答再按「检索有没有排到」拆两支：这一支才是老表里没有位置的第三种失败模式。
        after_miss = [r for r in refused if not (r.rank and r.rank <= k)]
        metrics["refused_after_miss_rate"] = round(len(after_miss) / len(scored), 4)
        metrics["refused_despite_hit_rate"] = round(
            (len(refused) - len(after_miss)) / len(scored), 4
        )

        # 检索没排到、模型却照样给答案——比拒答更危险，因为表面看起来"答了"。
        # 端到端作答率会把这类算成成功，所以必须单列，否则那个数字会骗人。
        answered_after_miss = [
            r
            for r in scored
            if not (r.abstained or r.llm_abstained) and not (r.rank and r.rank <= k)
        ]
        metrics["answered_after_miss_rate"] = round(len(answered_after_miss) / len(scored), 4)

    scored_una = [r for r in unanswerable if r.abstained is not None]
    if scored_una:
        by_threshold = sum(r.abstained for r in scored_una)
        by_llm = sum(1 for r in scored_una if r.abstained is False and r.llm_abstained)
        hallucinated = sum(1 for r in scored_una if r.abstained is False and not r.llm_abstained)
        metrics["abstain_by_threshold"] = round(by_threshold / len(scored_una), 4)
        metrics["abstain_by_llm"] = round(by_llm / len(scored_una), 4)
        metrics["false_answer_rate"] = round(hallucinated / len(scored_una), 4)

    latencies = sorted(r.latency_ms for r in rows if r.latency_ms)
    if latencies:
        metrics["retrieval_p50_ms"] = round(latencies[len(latencies) // 2], 1)
        metrics["retrieval_p95_ms"] = round(
            latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))], 1
        )
    return metrics


def sweep_thresholds(rows: list[Row]) -> list[dict]:
    """用同一批 /search 结果离线套阈值。

    不能反复打 /query 来扫：MIN_SCORE 在服务端，改一次要重启一次，且每档都要重跑
    全部问答、白烧 token。服务端判拒答的逻辑就是「最高分 < MIN_SCORE」，离线套等价。
    """
    answerable = [r for r in rows if r.is_answerable]
    unanswerable = [r for r in rows if not r.is_answerable]
    table: list[dict] = []

    for step in range(19):  # 0.00 → 0.90，步长 0.05
        threshold = round(step * 0.05, 2)
        correct = sum(1 for r in unanswerable if r.top_score < threshold)
        false_abstain = sum(1 for r in answerable if r.top_score < threshold)
        refused = correct + false_abstain
        precision = correct / refused if refused else 0.0
        recall = correct / len(unanswerable) if unanswerable else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        table.append(
            {
                "min_score": threshold,
                "abstain_recall": round(recall, 4),
                "false_abstain_rate": (
                    round(false_abstain / len(answerable), 4) if answerable else 0.0
                ),
                "precision": round(precision, 4),
                "f1": round(f1, 4),
            }
        )
    return table


# --------------------------------------------------------------------- 报告
def _carry_attributions(path: Path) -> dict[str, str]:
    """把上一轮手写的归因按题号捞回来。

    每次跑都会重写 failures.md，手写的归因会被冲掉——而那一栏恰恰是唯一不能自动生成的
    东西（博客素材全在这儿）。所以写盘前先把旧的读一遍，有内容就按题号续上。
    """
    if not path.is_file():
        return {}
    carried: dict[str, str] = {}
    current: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("### "):
            current = line[4:].split(" ", 1)[0]
            continue
        if current and line.startswith("- 归因："):
            why = line[len("- 归因：") :].strip()
            if why and not why.startswith("_（"):
                carried[current] = why
            current = None
    return carried


def build_cases_csv(rows: list[Row], k: int) -> str:
    lines = ["id,type,rank_strict,rank_loose,top1,k,category,abstained,llm_abstained,question"]
    for row in rows:
        cells = [
            row.case.id,
            row.case.type,
            "" if row.rank_strict is None else str(row.rank_strict),
            "" if row.rank is None else str(row.rank),
            f"{row.top_score:.4f}",
            str(k),
            classify(row, k),
            "" if row.abstained is None else str(int(row.abstained)),
            "" if row.llm_abstained is None else str(int(row.llm_abstained)),
            row.case.question,
        ]
        lines.append(",".join('"' + c.replace('"', '""') + '"' for c in cells))
    return "\n".join(lines) + "\n"


def build_failures(rows: list[Row], k: int, carried: dict[str, str] | None = None) -> str:
    missed = [r for r in rows if r.is_answerable and not (r.rank and r.rank <= k)]
    # 只有「阈值没拦 + 模型也没拒」才算真失败；模型自己兜住的不该记在这儿
    false_answers = [
        r for r in rows if not r.is_answerable and r.abstained is False and not r.llm_abstained
    ]
    rescued = [
        r for r in rows if not r.is_answerable and r.abstained is False and r.llm_abstained
    ]
    carried = carried or {}

    lines = ["# 失败清单", ""]
    lines += [
        "> 「归类」列是按名次机械判定的，可复现；「归因」列是人写的判断，两者对不上就有一边错。",
        f"> 口径：hit@{k}，宽松（任一承载文档都算）。严格口径的差在 cases.csv 的 rank_strict 列。",
        "",
    ]

    if not missed and not false_answers:
        lines.append("本轮没有真正的失败样本。题库太小或题目太容易时也会这样，别急着高兴。")
        if rescued:
            lines += [
                "",
                f"> 另有 {len(rescued)} 道无答案题是**模型自己拒绝回答**兜住的（阈值没拦住）："
                + "、".join(r.case.id for r in rescued),
                "> 这不算失败，但说明你的 MIN_SCORE 在这批题上没起作用——阈值该重新标定。",
            ]
        return "\n".join(lines) + "\n"

    if missed:
        buckets: dict[str, list[Row]] = {}
        for row in missed:
            buckets.setdefault(classify(row, k), []).append(row)
        lines += [f"## 未命中（{len(missed)} 道，hit@{k} 分母内）", "", "按归类分布：", ""]
        for name, group in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
            lines.append(f"- **{name}**：{len(group)} 道（{'、'.join(r.case.id for r in group)}）")
        lines += [
            "",
            "> 「近截断」加两个名额就能救；「排名靠后」加名额是自欺欺人（噪声一起进 Prompt），"
            "只能靠 rerank；「仅宽松命中」是口径问题；「未进 top20」才是检索能力问题。",
            "",
        ]
        for row in missed:
            lines.append(f"### {row.case.id} {row.case.question}")
            lines.append("")
            if row.case.answer:
                lines.append(f"- 标准答案：{row.case.answer}")
            lines.append(f"- 归类：{classify(row, k)}")
            if row.strict_keys:
                lines.append(
                    "- 主承载：" + "、".join(f"`{d}:{c}`" for d, c in row.strict_keys)
                )
            alts = [k2 for k2 in row.gold_keys if k2 not in row.strict_keys]
            if alts:
                lines.append("- 替代承载：" + "、".join(f"`{d}:{c}`" for d, c in alts))
            if not row.gold_keys:
                lines.append("- 应命中：_（gold 未解析，见上方告警）_")
            actual = "、".join(
                f"`{Path(str(h.get('file'))).name}#{h.get('chunk_index')}`"
                f"(p{h.get('page')},{float(h.get('score') or 0):.3f})"
                for h in row.hits[:5]
            )
            lines.append(f"- 实际 top5：{actual or '_无命中_'}")
            lines.append("- 归因：" + carried.get(row.case.id, "_（待填）_"))
            lines.append("")

    if false_answers:
        lines += [
            f"## 该拒答却回答了（{len(false_answers)} 道）",
            "",
            "> 这类比未命中更危险：文档里压根没有这个事实，模型却给了答案。",
            "> 要么阈值太低，要么题目没做到「主题沾边、事实不存在」。",
            "",
        ]
        for row in false_answers:
            lines.append(f"### {row.case.id} {row.case.question}")
            lines.append("")
            lines.append(f"- top1 分数：{row.top_score:.3f}")
            lines.append(f"- 实际回答：{(row.answer or '')[:160]}")
            lines.append(
                "- 归因："
                + carried.get(
                    row.case.id, "_（阈值过低 / 题目不和主题沾边 / 检索到了别的段落）_"
                )
            )
            lines.append("")

    if rescued:
        lines += [
            f"## 阈值没拦住、模型自己兜住了（{len(rescued)} 道）",
            "",
            "> 这些不算失败，但暴露出 MIN_SCORE 在这批题上形同虚设。",
            "> 面试时这是个能讲的点：拒答有「阈值」和「模型自主」两条路，各自失效方式不同。",
            "",
        ]
        for row in rescued:
            lines.append(
                f"- **{row.case.id}** {row.case.question} — top1={row.top_score:.3f}，"
                f"模型回答开头：{(row.answer or '')[:40]}…"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def build_citation_review(rows: list[Row], sample: int = 20, seed: int = 7) -> str:
    pool = [r for r in rows if r.answer is not None]
    random.Random(seed).shuffle(pool)
    picked = pool[:sample]

    lines = [
        "# 引用正确率抽检",
        "",
        "判定标准：引用片段是否**足以支撑**答案里对应的那句结论。三档，别打分：",
        "",
        "- 完全支持：只看这条片段就能得出该结论",
        "- 部分支持：片段相关，但结论比片段多走了一步",
        "- 不支持：片段和结论无关，或结论来自模型常识",
        "",
        f"抽了 {len(picked)} 条。README 里就写「{len(picked)} 条人工抽检，X 完全支持 / "
        f"Y 部分支持 / Z 不支持」，**不要换算成百分比**——20 个样本的 95% 置信区间有"
        "±20 个百分点，撑不住追问。",
        "",
        "---",
        "",
    ]
    for row in picked:
        lines.append(f"## {row.case.id} {row.case.question}")
        lines.append("")
        if row.case.answer:
            lines.append(f"**标准答案**：{row.case.answer}")
            lines.append("")
        lines.append(f"**模型回答**：{row.answer}")
        lines.append("")
        cited = [c for c in row.citations if c.get("used")] or row.citations
        if not cited:
            lines.append("_（本题拒答，无引用）_")
        for cite in cited[:4]:
            lines.append(
                f"- [{cite.get('index')}] {Path(str(cite.get('file'))).name} "
                f"第 {cite.get('page')} 页（score={cite.get('score')}）："
                f"{(cite.get('content') or '')[:100]}…"
            )
        lines.append("")
        lines.append("**判定**：_（完全支持 / 部分支持 / 不支持）_")
        lines.append("")
    return "\n".join(lines) + "\n"


def _w(text: str) -> int:
    """终端显示宽度：中日韩字符占两格。"""
    return sum(2 if ord(ch) > 0x2E80 else 1 for ch in text)


def _pad(text: str, width: int, align: str = "left") -> str:
    gap = max(0, width - _w(text))
    return text + " " * gap if align == "left" else " " * gap + text


def _cell(value, kind: str) -> str:
    """kind: pct / num / blank"""
    if value is None:
        return "—"
    if kind == "pct":
        return f"{value:.2%}"
    if kind == "num":
        return f"{value:.4f}"
    return str(value)


def print_report(rows: list[Row], metrics: dict, k: int, settings, hash_value: str) -> None:
    print("")
    print(
        f"配置指纹 {hash_value} | embedding={Path(str(settings.embedding_model)).name} "
        f"chunk={settings.chunk_size}/{settings.chunk_overlap} "
        f"MIN_SCORE={settings.min_score} top_k={settings.top_k} hit@{k} "
        f"mode={getattr(settings, 'retrieval_mode', 'vector')}"
    )
    line = "-" * 66
    print(line)
    print(_pad("口径", 28) + _pad(f"hit@{k}", 12, "right") + _pad("MRR", 12, "right")
          + _pad("分母", 8, "right"))
    print(line)
    table = [
        ("严格（只认主承载文档）", metrics.get(f"strict_hit@{k}"), metrics.get("strict_mrr"),
         metrics.get("strict_n")),
        ("宽松（任一承载文档都算）", metrics.get(f"loose_hit@{k}"), metrics.get("loose_mrr"),
         metrics.get("loose_n")),
        ("端到端（命中且模型作答）", metrics.get("e2e_answer_rate"), None,
         metrics.get("e2e_n")),
    ]
    for name, hit, mrr, n in table:
        print(
            _pad(name, 28)
            + _pad(_cell(hit, "pct"), 12, "right")
            + _pad(_cell(mrr, "num") if name != "端到端（命中且模型作答）" else "—", 12, "right")
            + _pad(_cell(n, "plain") if n is not None else "—", 8, "right")
        )
    print(line)
    gap = metrics.get("caliber_gap")
    if gap is not None:
        print(_pad("口径差额（宽松 − 严格）", 28) + f"{gap:+.2%}  ← 这一档是标注口径，不是检索能力")

    if "answerable_refused_rate" in metrics:
        print("")
        print("拒答分解（分母 = 有答案题里打过 /query 的）")
        print(line)
        items = [
            ("模型拒答合计", metrics.get("answerable_refused_rate")),
            ("　其中：检索没排到 → 拒答", metrics.get("refused_after_miss_rate")),
            ("　其中：排到了仍拒答", metrics.get("refused_despite_hit_rate")),
            ("检索没排到 → 仍硬答", metrics.get("answered_after_miss_rate")),
        ]
        for name, value in items:
            print(_pad(name, 28) + _pad(_cell(value, "pct"), 12, "right"))
        print(_pad("三支之和 = 检索未命中率", 28)
              + _pad(f"{(metrics.get('refused_after_miss_rate') or 0) + (metrics.get('answered_after_miss_rate') or 0):.2%}", 12, "right"))
        print(line)

    if "abstain_by_threshold" in metrics:
        print("")
        print("无答案题（分母 = 10）")
        print(line)
        print(_pad("拒答·阈值路径", 28) + _pad(_cell(metrics.get("abstain_by_threshold"), "pct"), 12, "right"))
        print(_pad("拒答·模型路径", 28) + _pad(_cell(metrics.get("abstain_by_llm"), "pct"), 12, "right")
              + "  启发式判定")
        print(_pad("真幻觉率", 28) + _pad(_cell(metrics.get("false_answer_rate"), "pct"), 12, "right")
              + "  两条路都没拦住")
        print(line)

    if "retrieval_p50_ms" in metrics:
        print(
            _pad("检索延迟 p50 / p95", 28)
            + f"{metrics['retrieval_p50_ms']:.0f} / {metrics['retrieval_p95_ms']:.0f} ms"
        )
        print(line)


# --------------------------------------------------------------------- 入口
def main() -> int:
    parser = argparse.ArgumentParser(description="RAG 评测")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--questions", default=str(Path(__file__).parent / "questions.yaml"))
    parser.add_argument("--k", type=int, default=6, help="hit@k 的 k，默认 6（=服务端 top_k）")
    parser.add_argument("--retrieve-k", type=int, default=20, help="/search 取多少条算 rank")
    parser.add_argument("--sweep", action="store_true", help="扫 MIN_SCORE，输出 F1 曲线")
    parser.add_argument("--no-query", action="store_true", help="不调 /query（省 token）")
    parser.add_argument("--set-baseline", action="store_true", help="把本轮结果写成 eval/baseline.json")
    parser.add_argument(
        "--caveat",
        action="append",
        default=[],
        help="给本轮结果打一个保留说明（可重复），会随 json 一起存下来，diff 时打印",
    )
    parser.add_argument(
        "--chroma-dir",
        default=None,
        help="覆盖向量库路径（默认读配置；在 WSL 的 9P 挂载上直接开库会卡住，可指向本地副本）",
    )
    parser.add_argument("--out-dir", default=str(Path(__file__).parent / "results"))
    args = parser.parse_args()

    settings = get_settings()
    cases = load_cases(Path(args.questions))
    if not cases:
        raise SystemExit("题库为空，先按 eval/README.md 填 eval/questions.yaml")

    hash_value = config_hash(settings, args.retrieve_k, args.k)
    rows_all = load_chunks(args.chroma_dir or settings.chroma_dir, settings.collection_name)
    docs = len({str(m.get("doc_id") or "") for m, _ in rows_all})
    print(f"库：{docs} 篇 / {len(rows_all)} 切片 | 题：{len(cases)} 道 | 指纹 {hash_value}")
    health = check_server(args.base_url, settings.api_key, settings)
    if health.get("chunks") and int(health["chunks"]) != len(rows_all):
        print(
            f"警告：服务端 {health['chunks']} 片，本地库 {len(rows_all)} 片。"
            "多半是入库后没重启服务，锚点反查和检索打的是两份索引",
            file=sys.stderr,
        )
    if args.k != settings.top_k:
        print(
            f"注意：hit@{args.k} 与服务端 top_k={settings.top_k} 不一致，"
            "端到端指标会失真（模型看到的是 top_k 条，不是 k 条）",
            file=sys.stderr,
        )

    if docs < 5 or len(rows_all) < 100:
        print(
            f"警告：只有 {docs} 篇 / {len(rows_all)} 切片。这个规模上 hit@{args.k} 没有区分度"
            f"（前 {args.k} 名已占全库 {args.k / max(len(rows_all), 1):.0%}），别写进简历。",
            file=sys.stderr,
        )

    with_query = not (args.no_query or args.sweep)
    print(f"采集：{'/search + /query' if with_query else '/search only'} @ {args.base_url}")
    rows = collect(args.base_url, cases, rows_all, args.retrieve_k, settings.api_key, with_query)

    metrics = compute_metrics(rows, args.k)
    print_report(rows, metrics, args.k, settings, hash_value)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d")

    payload: dict = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "config_hash": hash_value,
        "base_url": args.base_url,
        "config": {
            "embedding_model": settings.embedding_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "min_score": settings.min_score,
            "top_k": settings.top_k,
            "retrieval_mode": getattr(settings, "retrieval_mode", "vector"),
            "collection": settings.collection_name,
            "k": args.k,
            "retrieve_k": args.retrieve_k,
        },
        "corpus": {"documents": docs, "chunks": len(rows_all)},
        "counts": {
            "answerable": sum(1 for r in rows if r.is_answerable),
            "unanswerable": sum(1 for r in rows if not r.is_answerable),
            "gold_resolved": sum(len(r.gold_keys) for r in rows),
            "gold_strict_resolved": sum(len(r.strict_keys) for r in rows),
            "gold_unresolved": sum(len(r.unresolved) for r in rows),
            "alternate_golds": sum(1 for c in cases for g in c.golds if g.alternate),
        },
        "metrics": metrics,
        "caveats": args.caveat,
        "cases": [
            {
                "id": r.case.id,
                "type": r.case.type,
                "rank_strict": r.rank_strict,
                "rank_loose": r.rank,
                "top1": round(r.top_score, 4),
                "category": classify(r, args.k),
                "abstained": r.abstained,
            }
            for r in rows
        ],
    }

    if args.sweep:
        table = sweep_thresholds(rows)
        payload["threshold_sweep"] = table
        best = max(table, key=lambda item: item["f1"])
        print("")
        print(f"{'MIN_SCORE':>10}{'阈值拒答率':>12}{'误拒率':>10}{'精确率':>10}{'F1':>8}")
        print("-" * 52)
        for item in table:
            flag = "  ← 最优" if item is best else ""
            print(
                f"{item['min_score']:>10.2f}{item['abstain_recall']:>12.2%}"
                f"{item['false_abstain_rate']:>10.2%}{item['precision']:>10.2%}"
                f"{item['f1']:>8.3f}{flag}"
            )
        print("")
        print(
            f"建议 MIN_SCORE={best['min_score']}（F1={best['f1']}，阈值拒答率 "
            f"{best['abstain_recall']:.2%}，误拒率 {best['false_abstain_rate']:.2%}）"
        )
        print("改之前先想清楚代价：调高换来更少胡说，代价是该答的也答不出来。")

    named = out_dir / f"{stamp}-{hash_value}.json"
    named.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    latest = out_dir / "latest.json"
    latest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "cases.csv").write_text(build_cases_csv(rows, args.k), encoding="utf-8-sig")

    failures_path = out_dir / "failures.md"
    carried = _carry_attributions(failures_path)
    failures_path.write_text(build_failures(rows, args.k, carried), encoding="utf-8")
    if carried:
        print(f"\n沿用上一轮手写的归因：{len(carried)} 条（按题号匹配）")

    review = out_dir / "citation_review.md"
    if with_query and not review.exists():
        review.write_text(build_citation_review(rows), encoding="utf-8")
        print(f"\n引用抽检样本已生成：{review}（判定完再把结果写进 README）")

    baseline_path = Path(__file__).parent / "baseline.json"
    if args.set_baseline:
        baseline_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nbaseline 已更新：{baseline_path}")

    print(f"\n结果：{named}")
    print(f"失败清单：{failures_path}")
    print(f"逐题归类：{out_dir / 'cases.csv'}")

    if baseline_path.is_file() and not args.set_baseline:
        try:
            from diff_eval import diff_payloads, format_diff, has_regression
        except ImportError:
            print("（找不到 scripts/diff_eval.py，跳过 baseline 对比）")
        else:
            base = json.loads(baseline_path.read_text(encoding="utf-8"))
            print(format_diff(diff_payloads(base, payload), base, payload))
            if has_regression(diff_payloads(base, payload)):
                print("⚠ 有指标倒退，见上表。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
