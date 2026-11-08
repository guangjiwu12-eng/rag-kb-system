#!/usr/bin/env python3
"""给定一句原文，列出全库所有「承载这句话」的切片——补标注时用它，别凭印象。

为什么需要它
------------
「同一句话出现在两部标准里」这种判断，凭记忆做的准确率很低。实测过一轮：
11 道未命中里自认为是「跨标准同义条文」的 5 道，回库核对后 3 道成立、2 道不成立
（其中 1 道是同一部标准换了章节，另 1 道是同主题但不是同一条规定，数字都不一样）。

把「同主题」当成「同一条规定」标成 alternate，等于放宽口径偷好看。这个脚本的作用
是让每条 alternate 都有原文出处，而不是有印象出处。

用法
----
    # 直接给句子
    python scripts/find_carriers.py --phrase "不应有遮挡物"

    # 从题库里某道题的现有锚点出发，找它的其他承载文档
    python scripts/find_carriers.py --question n9

    # 顺手吐出可以直接粘进 questions.yaml 的 gold 块
    python scripts/find_carriers.py --phrase "压力表量程" --yaml

注意
----
读取的是**切片文本**，不是原文。所以要贴在 anchor 里的字符串必须是这里打印出来的
原样子串（含空格换行都会被 norm 掉，但字不能改）——改一个字就反查不到。

在 WSL 的 9P 挂载上直接开 Chroma 会卡死，请用本地副本（--chroma-dir 指向快照目录）。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_WS = re.compile(r"\s+")
_BREAK = re.compile(r"[。；;\n]")


def norm(text: str) -> str:
    return _WS.sub("", text or "")


def load_all(chroma_dir: str, collection: str) -> list[tuple[dict, str]]:
    import chromadb

    client = chromadb.PersistentClient(path=chroma_dir)
    try:
        coll = client.get_collection(collection)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"读不到集合 {collection} @ {chroma_dir}：{exc}") from exc
    data = coll.get(include=["documents", "metadatas"])
    return [
        (dict(meta), text or "")
        for text, meta in zip(data.get("documents") or [], data.get("metadatas") or [], strict=False)
        if meta is not None
    ]


def sentence_around(text: str, phrase: str, context: int) -> str:
    """取出包含 phrase 的那一句（按中文句读切），失败则截上下文。

    贴进 anchor 的片段越长越唯一，但太长会把整段复制进题库。一句通常刚好。
    """
    flat = _WS.sub(" ", text)
    target = _WS.sub(" ", phrase)
    pos = flat.find(target)
    if pos < 0:
        squeezed = norm(text)
        q = norm(phrase)
        idx = squeezed.find(q)
        if idx < 0:
            return flat[: context * 2]
        return squeezed[max(0, idx - context) : idx + len(q) + context]

    start = 0
    for match in _BREAK.finditer(flat, 0, pos):
        start = match.end()
    end_match = _BREAK.search(flat, pos + len(target))
    end = end_match.end() if end_match else min(len(flat), pos + len(target) + context)
    piece = flat[start:end].strip(" 。；;")
    if len(piece) > 160:
        piece = flat[max(0, pos - context) : pos + len(target) + context]
    return piece


def find(rows: list[tuple[dict, str]], phrase: str, context: int) -> list[dict]:
    q = norm(phrase)
    if len(q) < 2:
        raise SystemExit("phrase 太短，至少 2 个字")
    hits = []
    for meta, text in rows:
        if q in norm(text):
            hits.append(
                {
                    "doc": Path(str(meta.get("source") or "")).name,
                    "chunk_index": int(meta.get("chunk_index") or 0),
                    "page": meta.get("page"),
                    "doc_id": str(meta.get("doc_id") or ""),
                    "snippet": sentence_around(text, phrase, context),
                }
            )
    hits.sort(key=lambda h: (h["doc"], h["chunk_index"]))
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description="反查某句话在库里的全部承载切片")
    parser.add_argument("--phrase", action="append", default=[], help="要找的原文，可重复")
    parser.add_argument("--question", help="题库里的题号，用它现有的 gold 锚点去找")
    parser.add_argument("--questions", default=str(ROOT / "eval" / "questions.yaml"))
    parser.add_argument("--chroma-dir", default=None, help="向量库路径（9P 上会卡死，用本地副本）")
    parser.add_argument("--collection", default="kb_documents")
    parser.add_argument("--context", type=int, default=80, help="句子切不出来时截多少字上下文")
    parser.add_argument("--yaml", action="store_true", help="输出可粘贴的 gold 块（默认 alternate: true）")
    parser.add_argument("--primary", action="store_true", help="和 --yaml 一起用：不写 alternate")
    args = parser.parse_args()

    from app.config import get_settings

    settings = get_settings()
    phrases = list(args.phrase)

    if args.question:
        import yaml

        items = yaml.safe_load(Path(args.questions).read_text(encoding="utf-8")) or []
        picked = [i for i in items if str(i.get("id")) == args.question]
        if not picked:
            raise SystemExit(f"题库里没有 {args.question}")
        for gold in picked[0].get("gold") or []:
            if gold.get("anchor"):
                phrases.append(str(gold["anchor"]))
                print(f"# {args.question} 现有锚点：[{gold.get('doc')} p{gold.get('page')}]")
        if not phrases:
            raise SystemExit(f"{args.question} 没有 gold 锚点")

    if not phrases:
        raise SystemExit("给一个 --phrase 或 --question")

    rows = load_all(args.chroma_dir or settings.chroma_dir, args.collection)
    print(f"# 库：{len(rows)} 切片 / {len({str(m.get('doc_id')) for m, _ in rows})} 篇")
    print("# 核对要点：新候选必须**承载同一条规定**（数值、适用对象都一致），")
    print("# 只是主题相近不算——那是检索该区分开的干扰项。\n")

    for phrase in phrases:
        hits = find(rows, phrase, args.context)
        print(f"## 「{phrase[:60]}」→ {len(hits)} 处")
        if not hits:
            print("   （零命中）\n")
            continue
        for idx, hit in enumerate(hits, start=1):
            print(
                f"  [{idx}] {hit['doc']}  chunk={hit['chunk_index']}  p{hit['page']}  "
                f"doc_id={hit['doc_id']}"
            )
            print(f"      {hit['snippet']}")
        print("")
        if args.yaml:
            for hit in hits:
                lines = [
                    f"    - doc: {hit['doc']}",
                    f"      page: {hit['page']}",
                    f"      anchor: {hit['snippet']}",
                ]
                if not args.primary:
                    lines.append("      alternate: true   # 同一条规定的另一承载文档，仅宽松口径计入")
                print("\n".join(lines))
            print("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
