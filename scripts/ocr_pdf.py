#!/usr/bin/env python3
"""扫描版 PDF → 带文字层的 PDF：扩库前把「稀疏 / 无效」的文档救回来。

为什么不是「OCR 成 txt」
------------------------
体检表的核心指标是**每页切片数**（`chunks / pages`）。OCR 成一份 txt 之后整篇只有
「一页」，片/页 这个指标直接失效，`data/` 里也没法再按页定位。gold 锚点里的 `page`
同样依赖真实页码——标 page 的前提是文档本身有页。

所以这里做的是「重排文字层」：逐页 OCR，把识别出的每一行按原坐标写回一张同尺寸的
新页面，文字用 `render_mode=3`（不可见）插入。产出的 PDF：
- 页数与原文件一致 → 体检表的片/页 仍然成立
- 文字层可被 pypdf 正常抽取 → 入库、切片、锚点反查全部照常
- 不嵌图片 → 文件从几十 MB 掉到几百 KB

判读
----
OCR 一定会带噪（`(P`、`S`、`f)` 这类碎片）。它们不影响检索，但**不要拿 OCR 文本当
标准答案抄**——题目和 gold 锚点要从入库后的 chunk 里复制，这样锚点才是唯一可匹配的。

用法
----
    python scripts/ocr_pdf.py --src data/raw_scan --dst data/inbox
    python scripts/ocr_pdf.py --src data/raw_scan --dst data/inbox --workers 4
    python scripts/ocr_pdf.py --src a.pdf --dst data/inbox --dpi 200   # 单文件试跑

依赖：pip install pymupdf rapidocr-onnxruntime
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

try:
    import pymupdf
except ImportError:  # pragma: no cover
    raise SystemExit("需要 PyMuPDF：pip install pymupdf") from None

DPI = 150
# 行间距不足一行高时合并成一段，避免把一句话拆成十几条碎片
LINE_GAP_RATIO = 0.6
INVISIBLE = 3  # PyMuPDF render_mode：3 = 不填充也不描边，文字不可见但可被抽取


def ocr_page(job: tuple[str, int, int]) -> tuple[str, int, list[tuple[float, float, float, str]]]:
    """单页 OCR：返回 (文件, 页码, [(x, y, 行高, 文本)])。在子进程里跑，引擎进程内复用。"""
    from rapidocr_onnxruntime import RapidOCR

    path, page_no, dpi = job
    engine = globals().get("_ENGINE")
    if engine is None:  # 进程内只初始化一次，否则每页都要重新加载模型
        engine = RapidOCR()
        globals()["_ENGINE"] = engine

    with pymupdf.open(path) as doc:
        page = doc[page_no]
        pixmap = page.get_pixmap(dpi=dpi)
        result, _ = engine(pixmap.tobytes("png"), use_cls=False)

    lines: list[tuple[float, float, float, str]] = []
    for box, text, _score in result or []:
        text = (text or "").strip()
        if not text:
            continue
        xs = [point[0] for point in box]
        ys = [point[1] for point in box]
        lines.append((min(xs), min(ys), max(ys) - min(ys), text))
    # 按行读取：先纵后横。OCR 框本身是无序的，不排序抽出来的文字会是乱的
    lines.sort(key=lambda item: (round(item[1] / 10), item[0]))
    return path, page_no, [(x, y, h, t) for x, y, h, t in lines]


def build_pdf(
    pages: list[tuple[int, list[tuple[float, float, float, str]]]], src: Path, dst: Path, dpi: int
) -> int:
    """把 OCR 结果写成同尺寸、同页数的不可见文字层 PDF，返回写入的字符数。"""
    scale = 72.0 / dpi  # 像素坐标 → PDF 点
    out = pymupdf.open()
    written = 0

    with pymupdf.open(src) as doc:
        for page_no, lines in sorted(pages):
            src_page = doc[page_no]
            page = out.new_page(width=src_page.rect.width, height=src_page.rect.height)

            for x, y, height, text in lines:
                font_size = max(4.0, min(11.0, height * scale * 0.85))
                page.insert_text(
                    (x * scale, (y + height * 0.8) * scale),
                    text,
                    fontname="china-s",  # PyMuPDF 内置 CJK 字体，带 ToUnicode，pypdf 能读
                    fontsize=font_size,
                    render_mode=INVISIBLE,
                )
                written += len(text)

    dst.parent.mkdir(parents=True, exist_ok=True)
    out.save(dst, garbage=4, deflate=True)
    out.close()
    return written


def convert(src: Path, dst: Path, workers: int, dpi: int) -> None:
    with pymupdf.open(src) as doc:
        total = doc.page_count
    started = time.perf_counter()
    jobs = [(str(src), page_no, dpi) for page_no in range(total)]

    results: list[tuple[int, list[tuple[float, float, float, str]]]] = []
    if workers <= 1:
        for index, job in enumerate(jobs, start=1):
            _, page_no, lines = ocr_page(job)
            results.append((page_no, lines))
            print(f"    {index}/{total} 页", end="\r", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for index, (_, page_no, lines) in enumerate(pool.map(ocr_page, jobs), start=1):
                results.append((page_no, lines))
                print(f"    {index}/{total} 页", end="\r", flush=True)

    chars = build_pdf(results, src, dst, dpi)
    elapsed = time.perf_counter() - started
    print(
        f"    -> {dst.name}  {total} 页 / {chars} 字 / {elapsed:.0f}s "
        f"({elapsed / max(total, 1):.1f}s 每页，{dst.stat().st_size // 1024} KB)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="扫描版 PDF 转带文字层 PDF")
    parser.add_argument("--src", required=True, help="源 PDF 或目录")
    parser.add_argument("--dst", required=True, help="输出目录")
    parser.add_argument("--workers", type=int, default=1, help="并行进程数（CPU 核数的一半左右）")
    parser.add_argument("--dpi", type=int, default=DPI)
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    files = [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() == ".pdf")
    if not files:
        raise SystemExit(f"{src} 下没有 PDF")

    failures = 0
    for path in files:
        print(f"[{path.name}]")
        try:
            convert(path, dst / path.name, args.workers, args.dpi)
        except Exception as exc:  # noqa: BLE001 - 单篇失败不该中断整批
            print(f"    ! 失败：{exc}", file=sys.stderr)
            failures += 1

    print(f"\n完成 {len(files) - failures}/{len(files)} 篇 -> {dst}")
    print("入库前先起服务，再跑体检：python scripts/ingest_dir.py --dir data/inbox")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
