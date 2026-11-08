#!/usr/bin/env python3
"""纯文本 → 分页 PDF：把「只有正文、没有版面」的语料变成能进库的文档。

为什么需要它
------------
知识库的体检表按**每页切片数**判断文档质量，gold 锚点也记 `page`。一段从网页抓下来
的规范全文如果直接存成 .txt，入库后整篇只有「一页」，这两个东西同时失效。

所以这里把正文按固定版心重新排版：A4、每页约 1100 字、页脚带页码。页码是**排版页**，
不是纸质标准的原书页码——标注 gold 时以入库后的 `page` 元数据为准。

用法
----
    python scripts/build_corpus_pdf.py --src data/sources --dst data/inbox
    python scripts/build_corpus_pdf.py --src a.txt --dst data/inbox --chars-per-page 900

依赖：pip install pymupdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import pymupdf
except ImportError:  # pragma: no cover
    raise SystemExit("需要 PyMuPDF：pip install pymupdf") from None

FONT = "china-s"  # PyMuPDF 内置中文字体，带 ToUnicode，pypdf / PyPDFLoader 都能抽
FONT_SIZE = 10.5
LINE_HEIGHT = 17.0
MARGIN_X = 56.0
MARGIN_Y = 56.0
PAGE_W, PAGE_H = 595.0, 842.0  # A4，单位 pt
CHARS_PER_PAGE = 1100


SENTENCE_ENDS = "。！？!?"
CLAUSE_ENDS = "；，、：）】》,;:"


def wrap(paragraph: str, chars_per_line: int) -> list[str]:
    """折行：优先句末，其次逗号，最后才硬切。

    硬折行会把句子切成「在执行中 / ，除执行本标准外」这种残句：切片的第二级分隔符正是
    `\\n`，于是每个残句都成独立切分点，chunk 从句子中间开始、在句子中间结束，命中后拿到
    的片段读不通。所以先往回找 60 字以内的句末标点，再找 30 字以内的逗号。
    """
    remaining = paragraph.strip()
    lines: list[str] = []
    while len(remaining) > chars_per_line:
        cut = 0
        for probe in range(chars_per_line, max(chars_per_line - 60, 0), -1):
            if remaining[probe - 1] in SENTENCE_ENDS:
                cut = probe
                break
        if not cut:
            for probe in range(chars_per_line, max(chars_per_line - 30, 0), -1):
                if remaining[probe - 1] in CLAUSE_ENDS:
                    cut = probe
                    break
        if not cut:
            cut = chars_per_line
        lines.append(remaining[:cut])
        remaining = remaining[cut:]
    if remaining:
        lines.append(remaining)
    return lines or [""]


def build(src: Path, dst: Path, chars_per_page: int) -> tuple[int, int]:
    text = src.read_text(encoding="utf-8", errors="replace")
    paragraphs = [line.strip() for line in text.splitlines() if line.strip()]

    usable_w = PAGE_W - 2 * MARGIN_X
    usable_h = PAGE_H - 2 * MARGIN_Y - 24  # 留出页脚
    chars_per_line = max(20, int(usable_w / FONT_SIZE))
    lines_per_page = max(10, int(usable_h / LINE_HEIGHT))

    # 每行多少字 → 每页多少行，两者对齐到 chars_per_page
    chars_per_line = min(chars_per_line, max(20, chars_per_page // lines_per_page))
    lines_per_page = max(10, chars_per_page // chars_per_line)

    lines: list[str] = []
    for paragraph in paragraphs:
        lines.extend(wrap(paragraph, chars_per_line))

    doc = pymupdf.open()
    page_no = 0
    for start in range(0, len(lines), lines_per_page):
        page_no += 1
        page = doc.new_page(width=PAGE_W, height=PAGE_H)
        y = MARGIN_Y + FONT_SIZE
        for line in lines[start : start + lines_per_page]:
            page.insert_text((MARGIN_X, y), line, fontname=FONT, fontsize=FONT_SIZE)
            y += LINE_HEIGHT
        page.insert_text(
            (MARGIN_X, PAGE_H - 32),
            f"第 {page_no} 页",
            fontname=FONT,
            fontsize=8,
            color=(0.45, 0.45, 0.45),
        )

    dst.parent.mkdir(parents=True, exist_ok=True)
    doc.save(dst, garbage=4, deflate=True)
    doc.close()
    return page_no, len("".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="纯文本转分页 PDF（扩库语料用）")
    parser.add_argument("--src", required=True, help="txt 文件或目录")
    parser.add_argument("--dst", required=True, help="输出目录")
    parser.add_argument("--chars-per-page", type=int, default=CHARS_PER_PAGE)
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    files = [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() == ".txt")
    if not files:
        raise SystemExit(f"{src} 下没有 txt")

    total_pages = 0
    for path in files:
        pages, chars = build(path, dst / f"{path.stem}.pdf", args.chars_per_page)
        total_pages += pages
        print(f"{path.name} -> {pages} 页 / {chars} 字（约 {chars // max(pages, 1)} 字每页）")
    print(f"\n共 {len(files)} 篇 / {total_pages} 页 -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
