#!/usr/bin/env python3
"""全量重建向量库：让「跨标准同义条文」真正只剩一份。

为什么单篇上传做不到这件事
--------------------------
`/documents` 一次只看一篇文档。可「同一句话在两部标准里各写一遍」是跨文档现象：
GB50166 与 GB51309 都写了「应有不少于设备总数 1% 的备品」。分两次上传时，
第二篇看不到第一篇的切片，两份就都留在库里，检索时互相挤占 Top-K，
而 gold 只标了一部标准——命中另一份会被判成未命中。

这个脚本先把整个语料切成片，在**全语料范围**内做去重，再一次性写进一个新集合。

保留溯源，不删出处
------------------
重复的切片不会静默消失：正本切片的 metadata 里会记 `also_in`（同一段文字还出现在
哪些文件、哪一页、哪一条），引用展示时可以照常告诉用户「该条文亦见于 GB50166」。
删的是重复向量，不是证据。

去重只认「完全一致」的文本
--------------------------
近似重复只报告、不合并。规范里 6.2.5~6.2.19 十九条巡查内容只差系统名，
相似度能到 0.8 以上，按相似度合并会把十九条不同的要求压成一条。

用法
----
    python scripts/rebuild_index.py --dry-run          # 只出体检报告，不写库（默认）
    python scripts/rebuild_index.py --apply            # 真的写库
    python scripts/rebuild_index.py --apply --collection kb_v2   # 写进新集合，旧库不动

报告：eval/results/corpus_audit.md
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import corpus  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.document_loader import chunk_documents, load_pages, sha256_bytes  # noqa: E402

SUPPORTED = {".pdf", ".txt"}


def collect_chunks(directories: list[Path], settings) -> list[dict]:
    """把目录里的文档全部切成片，带上 doc_id 与文件信息。

    支持传多个目录：真实库里的文件就分两处（data/实践日志.pdf 和 data/inbox/ 里的规范），
    只扫一处会漏掉另一半，去重也就少了一半的比对对象。
    """
    files: list[Path] = []
    for directory in directories:
        files.extend(sorted(p for p in directory.iterdir() if p.suffix.lower() in SUPPORTED))
    if not files:
        raise SystemExit(f"{' / '.join(str(d) for d in directories)} 下没有 PDF/TXT")

    items: list[dict] = []
    for path in files:
        data = path.read_bytes()
        doc_id = sha256_bytes(data)[:16]
        pages = load_pages(path, path.suffix.lower().lstrip("."))
        chunks = chunk_documents(pages, settings)
        for chunk in chunks:
            items.append(
                {
                    "doc_id": doc_id,
                    "file": path.name,
                    "chunk_index": chunk.metadata["chunk_index"],
                    "page": int(chunk.metadata.get("page") or 0) + 1,
                    "page_printed": int(chunk.metadata.get("page_printed") or 0),
                    "article": chunk.metadata.get("article") or "",
                    "section_type": chunk.metadata.get("section_type") or "main",
                    "section_path": chunk.metadata.get("section_path") or "",
                    "chunk_mode": chunk.metadata.get("chunk_mode") or "",
                    "text": chunk.page_content,
                }
            )
    return items


def group_duplicates(items: list[dict]) -> dict[str, list[int]]:
    """按指纹分组，只返回出现多于一次的组。"""
    groups: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(items):
        key = corpus.fingerprint(item["text"])
        if len(key) >= 8:
            groups[key].append(index)
    return {key: ids for key, ids in groups.items() if len(ids) > 1}


def choose_canonical(items: list[dict], ids: list[int]) -> int:
    """一组重复里挑正本：优先正文条文，其次更长的，最后按文件名字典序。"""
    return sorted(
        ids,
        key=lambda i: (
            items[i]["section_type"] != "main",
            -len(items[i]["text"]),
            items[i]["file"],
            items[i]["page"],
        ),
    )[0]


def build_plan(items: list[dict]) -> list[dict]:
    """就地标注 dup_of / also_in，返回需要写入的切片列表。"""
    groups = group_duplicates(items)
    drop: set[int] = set()

    for ids in groups.values():
        keeper = choose_canonical(items, ids)
        aliases = []
        for other in ids:
            if other == keeper:
                continue
            drop.add(other)
            src = items[other]
            aliases.append(f"{src['file']}:{src['page']}:{src['article'] or '-'}")
        items[keeper]["also_in"] = "；".join(dict.fromkeys(aliases))

    kept = [item for index, item in enumerate(items) if index not in drop]
    # 去重后按文档重新编号，保证 chunk_index 连续
    counters: Counter[str] = Counter()
    for item in kept:
        item["chunk_index"] = counters[item["doc_id"]]
        counters[item["doc_id"]] += 1
    return kept


def audit_report(
    items: list[dict], kept: list[dict], duplicates: dict[str, list[int]], near_threshold: float
) -> str:
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        by_doc[item["file"]].append(item)

    cross_doc = [
        ids for ids in duplicates.values() if len({items[i]["file"] for i in ids}) > 1
    ]
    near = corpus.near_duplicate_pairs([item["text"] for item in kept], threshold=near_threshold)

    lines = [
        "# 语料体检报告",
        "",
        f"生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "## 切片构成",
        "",
        f"- 文档 {len(by_doc)} 篇，切片 {len(items)} 片，去重后 {len(kept)} 片（合并 {len(items) - len(kept)} 片）",
        f"- 跨文档完全重复的组：{len(cross_doc)}",
        "",
        "| 文件 | 页 | 切片 | 正文 | 条文说明 | 前置 | 均长 | 页码偏移 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, chunks in sorted(by_doc.items()):
        types = Counter(c["section_type"] for c in chunks)
        lens = [len(c["text"]) for c in chunks]
        pages = max((c["page"] for c in chunks), default=0)
        offset = {c["page_printed"] - c["page"] for c in chunks if c["page_printed"]}
        lines.append(
            f"| {name} | {pages} | {len(chunks)} | {types.get('main', 0)} | "
            f"{types.get('explanation', 0)} | {types.get('front', 0)} | "
            f"{sum(lens) // max(len(lens), 1)} | {offset.pop() if len(offset) == 1 else '不一致'} |"
        )

    lines += ["", "## 跨文档完全重复（去重时并入正本）", ""]
    if not cross_doc:
        lines.append("无。")
    for ids in cross_doc:
        keeper = choose_canonical(items, ids)
        lines.append(f"- 正本：{items[keeper]['file']} 第 {items[keeper]['page']} 页 "
                     f"第 {items[keeper]['article'] or '-'} 条")
        for other in ids:
            if other == keeper:
                continue
            lines.append(
                f"  - 并入：{items[other]['file']} 第 {items[other]['page']} 页 "
                f"第 {items[other]['article'] or '-'} 条"
            )
        lines.append(f"  - 原文：{items[keeper]['text'][:80]}")

    lines += ["", "## 近似重复（只报告，不合并）", ""]
    lines.append(
        f"相似度 ≥ {near_threshold:.2f} 且不是完全一致——多半是不同标准里的近义条文，需要人判断。"
    )
    lines.append("")
    if not near:
        lines.append("无。")
    for first, second, score in near[:40]:
        a, b = kept[first], kept[second]
        if a["file"] == b["file"]:
            continue
        lines.append(
            f"- {score:.2f}｜{a['file']} 第 {a['page']} 页 条{a['article'] or '-'} "
            f"≈ {b['file']} 第 {b['page']} 页 条{b['article'] or '-'}"
        )
    lines.append("")
    return "\n".join(lines)


def write_collection(kept: list[dict], settings) -> int:
    """把去重后的切片写进向量库（目标集合不存在则新建）。"""
    from langchain_core.documents import Document

    from app.llm_factory import build_embeddings
    from app.vector_store import VectorStore

    store = VectorStore(settings, build_embeddings(settings))
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for item in kept:
        by_doc[item["doc_id"]].append(item)

    written = 0
    started = perf_counter()
    total = len(kept)
    for doc_id, chunks in by_doc.items():
        documents = []
        for item in chunks:
            metadata = {
                "doc_id": doc_id,
                "source": item["file"],
                "chunk_index": item["chunk_index"],
                "page": item["page"],
                "page_printed": item["page_printed"],
                "article": item["article"],
                "section_type": item["section_type"],
                "section_path": item["section_path"],
                "chunk_mode": item["chunk_mode"],
                "also_in": item.get("also_in", ""),
                "added_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            }
            documents.append(Document(page_content=item["text"], metadata=metadata))
        ids = [f"{doc_id}:{item['chunk_index']}" for item in chunks]
        # 这一行必须在 upsert 之前打：本地 bge-m3 在 CPU 上是分钟级的事，
        # 卡住时如果只有「写完一篇才打印」，屏幕上就一直是空白，看不出是在算还是死了。
        print(f"  写入 {chunks[0]['file']}：{len(documents)} 片 …", file=sys.stderr, flush=True)
        store.upsert_document(documents, ids, doc_id, chunks[0]["file"])
        written += len(documents)
        elapsed = perf_counter() - started
        rate = written / elapsed if elapsed > 0 else 0.0
        eta = (total - written) / rate if rate > 0 else 0.0
        print(
            f"    完成 {written}/{total} 片｜{elapsed:.0f}s｜{rate:.1f} 片/秒｜剩余约 {eta:.0f}s",
            file=sys.stderr,
            flush=True,
        )
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="全量重建向量库（跨文档去重）")
    parser.add_argument("--dir", nargs="+", default=["data/inbox", "data"])
    parser.add_argument("--apply", action="store_true", help="真的写库；不加只出报告")
    parser.add_argument("--collection", default=None, help="目标集合名，默认沿用配置")
    parser.add_argument("--report", default="eval/results/corpus_audit.md")
    args = parser.parse_args()

    settings = get_settings()
    directories = [Path(item) if Path(item).is_absolute() else ROOT / item for item in args.dir]
    seen: set[str] = set()
    unique: list[Path] = []
    for directory in directories:
        key = str(directory.resolve()).casefold()
        if key not in seen:
            seen.add(key)
            unique.append(directory)

    items = collect_chunks(unique, settings)
    duplicates = group_duplicates(items)
    kept = build_plan(items)

    print(f"切片 {len(items)} → 去重后 {len(kept)}（合并 {len(items) - len(kept)}）")
    by_type = Counter(item["section_type"] for item in kept)
    print(f"正文 {by_type.get('main', 0)} / 条文说明 {by_type.get('explanation', 0)} / 前置 {by_type.get('front', 0)}")
    print(f"跨文档完全重复：{sum(1 for ids in duplicates.values() if len({items[i]['file'] for i in ids}) > 1)} 组")

    report = ROOT / args.report
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        audit_report(items, kept, duplicates, settings.dedupe_threshold), encoding="utf-8"
    )
    print(f"体检报告：{report}")

    if not args.apply:
        print("\n只出报告，没写库。确认无误后加 --apply。")
        print("建议先写到新集合：--apply --collection kb_v2，验收完再改配置切过去。")
        return 0

    if args.collection:
        settings = settings.model_copy(update={"collection_name": args.collection})
    written = write_collection(kept, settings)
    print(f"\n已写入集合 {settings.collection_name}：{written} 片")
    print("接下来：重启服务 → python eval/eval.py → python scripts/verify_citations.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
