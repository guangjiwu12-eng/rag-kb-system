#!/usr/bin/env python3
"""批量入库 + 切片体检：扩库时用，专门治「扫描件静默降级」这个坑。

为什么需要它
------------
PDF 能上传成功，不等于切片正常。图表型/扫描型 PDF 的文字层可能只有几百字，
800 字的切片口径切下去只剩三五个碎片，检索必然垮——但接口返回 201，你不会知道。
这份脚本把「每页切片数」摊在表上，低于阈值直接标红。

用法
----
    python scripts/ingest_dir.py                    # 入库 data/ 下的全部 PDF/TXT
    python scripts/ingest_dir.py --dir data/inbox   # 指定目录
    python scripts/ingest_dir.py --dry-run          # 只体检，不入库

判读
----
    每页切片数 < 1.0   → 稀疏：多半是扫描件或表格，建议 OCR 后重传，别放进知识库
    切片数 < 3         → 无效：这一篇几乎不会贡献召回，删掉
    正常文档一般在 1.5-6 片/页之间
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

try:
    import requests
except ImportError:  # pragma: no cover
    raise SystemExit("需要 requests：pip install requests") from None

SUPPORTED = {".pdf", ".txt"}
CHUNKS_PER_PAGE_WARN = 1.0


def doc_id_of(data: bytes) -> str:
    """与服务端一致的 doc_id 口径：sha256 前 16 位。"""
    return hashlib.sha256(data).hexdigest()[:16]


def existing_doc_ids(base_url: str, api_key: str | None) -> set[str]:
    headers = {"X-API-Key": api_key} if api_key else {}
    try:
        response = requests.get(f"{base_url}/api/v1/documents", headers=headers, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise SystemExit(f"读不到文档列表：{exc}") from exc
    return {item["doc_id"] for item in response.json().get("documents", [])}


def upload(base_url: str, path: Path, api_key: str | None) -> dict:
    headers = {"X-API-Key": api_key} if api_key else {}
    with path.open("rb") as handle:
        files = {"file": (path.name, handle, "application/octet-stream")}
        response = requests.post(
            f"{base_url}/api/v1/documents", files=files, headers=headers, timeout=600
        )
    if response.status_code >= 400:
        return {"error": f"{response.status_code} {response.text[:120]}"}
    return response.json()


def verdict(chunks: int, pages: int) -> tuple[str, str]:
    per_page = chunks / pages if pages else 0.0
    if chunks < 3:
        return "无效", f"只有 {chunks} 片，几乎不会贡献召回"
    if per_page < CHUNKS_PER_PAGE_WARN:
        return "稀疏", f"{per_page:.1f} 片/页，疑似扫描件或图表型 PDF"
    return "ok", f"{per_page:.1f} 片/页"


def main() -> int:
    parser = argparse.ArgumentParser(description="批量入库并体检切片质量")
    parser.add_argument("--dir", default="data")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--dry-run", action="store_true", help="只统计文件，不上传")
    args = parser.parse_args()

    root = Path(args.dir)
    if not root.is_dir():
        raise SystemExit(f"目录不存在：{root}")
    files = sorted(p for p in root.iterdir() if p.suffix.lower() in SUPPORTED)
    if not files:
        raise SystemExit(f"{root} 下没有 PDF/TXT")

    existing: set[str] = set()
    if not args.dry_run:
        existing = existing_doc_ids(args.base_url, args.api_key)
        print(f"服务端已有 {len(existing)} 篇；本次待处理 {len(files)} 个文件\n")

    rows: list[tuple[str, str, str, str, str]] = []
    total_chunks = 0
    for path in files:
        data = path.read_bytes()
        if args.dry_run:
            rows.append((path.name, "-", "-", "-", f"{len(data) // 1024} KB"))
            continue

        if doc_id_of(data) in existing:
            rows.append((path.name, "-", "-", "跳过", "内容已在库中"))
            continue

        result = upload(args.base_url, path, args.api_key)
        if "error" in result:
            rows.append((path.name, "-", "-", "失败", result["error"]))
            continue

        chunks, pages = int(result["chunks"]), int(result["pages"])
        total_chunks += chunks
        state, note = verdict(chunks, pages)
        rows.append((path.name, str(pages), str(chunks), state, note))

    width = max(len(r[0]) for r in rows)
    print(f"{'文件':<{width}}  {'页':>4} {'切片':>5}  {'状态':<5} 备注")
    print("-" * (width + 42))
    for name, pages, chunks, state, note in rows:
        flag = "" if state in ("ok", "-", "跳过") else "  <-- 处理一下"
        print(f"{name:<{width}}  {pages:>4} {chunks:>5}  {state:<5} {note}{flag}")

    if not args.dry_run:
        print("-" * (width + 42))
        print(f"本次新增切片：{total_chunks}")
        print("\n扩库目标：10-20 篇 / 300+ 切片。少于这个规模，hit@4 没有区分度。")
        print("入库完跑一次：python eval/eval.py --no-query")
    return 0


if __name__ == "__main__":
    sys.exit(main())
