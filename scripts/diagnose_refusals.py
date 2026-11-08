"""诊断「检索排到了、模型却拒答」的个案。

为什么要单独一个脚本：这类失败在汇总指标里只是一个百分比，但每一道的成因不同——
有的是答案只在条文说明里、有的是判据过严、有的干脆是 LLM 抖动。
不把原文和回答摆在一起看，就只能靠猜，而猜错的归因会把后面的优化全带偏。

用法（服务需先起）：
    python scripts/diagnose_refusals.py --out eval/results/refusals.md
    python scripts/diagnose_refusals.py --ids n10,n23 --dump-context
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import yaml

REFUSAL_MARKERS = ("无法回答", "没有提到", "未提及", "资料不足")

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with OPENER.open(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def load_cases(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    return [c for c in data if c.get("type") == "answerable"]


def main() -> None:
    parser = argparse.ArgumentParser(description="列出检索命中但模型拒答的个案")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--questions", default="eval/questions.yaml")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--ids", default="", help="只看指定题目，逗号分隔")
    parser.add_argument("--dump-context", action="store_true", help="同时打印喂给模型的片段")
    parser.add_argument("--out", default="", help="写出 markdown 报告")
    args = parser.parse_args()

    wanted = {s.strip() for s in args.ids.split(",") if s.strip()}
    cases = load_cases(Path(args.questions))
    if wanted:
        cases = [c for c in cases if c["id"] in wanted]

    lines: list[str] = ["# 命中仍拒答 · 逐题诊断", ""]
    refused: list[dict] = []

    for case in cases:
        question = case["question"]
        answer = post(f"{args.base_url}/api/v1/query", {"question": question, "top_k": args.top_k})
        text = answer.get("answer") or ""
        is_refused = any(marker in text for marker in REFUSAL_MARKERS)
        if not is_refused:
            continue

        refused.append(case)
        searched = post(
            f"{args.base_url}/api/v1/search", {"question": question, "top_k": args.top_k}
        )
        hits = searched.get("hits") or []

        lines.append(f"## {case['id']} {question}")
        lines.append("")
        lines.append(f"- 标准答案：{case.get('answer', '').strip()}")
        lines.append(f"- 模型回答：{text[:220]}")
        lines.append("")
        for hit in hits:
            label = f"{hit.get('file')} 第{hit.get('page')}页"
            if hit.get("article"):
                label += f" 第{hit['article']}条"
            if hit.get("section_type") == "explanation":
                label += "【条文说明】"
            lines.append(f"- 命中 {hit.get('index')}：{label}")
            if args.dump_context:
                body = (hit.get("content") or "").replace("\n", " ")
                lines.append(f"  > {body[:300]}")
        lines.append("")

    lines.insert(2, f"共 {len(refused)} 道：{', '.join(c['id'] for c in refused)}")
    lines.insert(3, "")

    report = "\n".join(lines)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"报告：{args.out}")
    print(report)


if __name__ == "__main__":
    main()
