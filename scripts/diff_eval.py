#!/usr/bin/env python3
"""对比两份评测结果，只报「哪几个指标动了」。

为什么需要它
------------
每次调参后重跑评测，输出是一屏数字。人会本能地找那个变好的数字，忽略变坏的。
把对比固化成工具，输出就只有「变动表」——没动的指标不出现在表里，
想自我安慰都没有空间。

用法
----
    python scripts/diff_eval.py                        # baseline.json vs results/latest.json
    python scripts/diff_eval.py --baseline a.json --current b.json
    python scripts/diff_eval.py --all                  # 连没变的也列
    python scripts/diff_eval.py --fail-on-regression    # 有倒退时退出码 1（可挂 CI）

方向约定
--------
- 越高越好：hit@k、MRR、端到端作答率、阈值拒答率、模型拒答率、精确率、F1
- 越低越好：误拒率、真幻觉率、各类拒答率（有答案题）、延迟
- 不判定：口径差额（caliber_gap）、分母（*_n）

把「有答案题的拒答率」判成越低越好是有前提的：那批题确实有答案。前提不成立时
这个判定会反过来骗你，所以它同时也被列在分母里给你看。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EPS = 1e-9

LOWER_IS_BETTER = (
    "false_abstain_rate",
    "false_answer_rate",
    "answerable_refused_rate",
    "refused_after_miss_rate",
    "refused_despite_hit_rate",
    "answered_after_miss_rate",
    "_ms",
)
HIGHER_IS_BETTER = (
    "hit@",
    "mrr",
    "e2e_answer_rate",
    "abstain_by_threshold",
    "abstain_by_llm",
    "precision",
    "abstain_recall",
    "f1",
)
NEUTRAL = ("caliber_gap", "_n", "min_score")


def direction(key: str) -> int:
    """+1 越高越好，-1 越低越好，0 不判定。无答案题的拒答率属于 +1。"""
    if any(token in key for token in NEUTRAL):
        return 0
    if any(token in key for token in LOWER_IS_BETTER):
        return -1
    if any(token in key for token in HIGHER_IS_BETTER):
        return 1
    return 0


def _is_pct(key: str) -> bool:
    return "rate" in key or "hit@" in key or "abstain_by" in key or "_recall" in key


def _is_ms(key: str) -> bool:
    return key.endswith("_ms")


def fmt(key: str, value) -> str:
    if value is None:
        return "—"
    if _is_ms(key):
        return f"{value:.0f}ms"
    if _is_pct(key):
        return f"{value:.2%}"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def fmt_delta(key: str, delta: float) -> str:
    if _is_ms(key):
        return f"{delta:+.0f}ms"
    if _is_pct(key):
        return f"{delta * 100:+.2f}pp"
    return f"{delta:+.4f}"


def diff_payloads(base: dict, current: dict) -> list[dict]:
    base_metrics = base.get("metrics") or {}
    cur_metrics = current.get("metrics") or {}
    keys = [k for k in cur_metrics if k in base_metrics]
    keys += [k for k in base_metrics if k not in cur_metrics]

    diffs: list[dict] = []
    for key in keys:
        before = base_metrics.get(key)
        after = cur_metrics.get(key)
        if before is None or after is None:
            if before is not after:
                diffs.append(
                    {
                        "key": key,
                        "before": before,
                        "after": after,
                        "delta": None,
                        "judgement": "新增/消失",
                        "regression": False,
                    }
                )
            continue
        delta = float(after) - float(before)
        if abs(delta) < EPS:
            continue
        dirn = direction(key)
        if dirn == 0:
            judgement, regression = "持平（不判定）", False
        elif delta * dirn > 0:
            judgement, regression = "↑ 变好", False
        else:
            judgement, regression = "↓ 变差", True
        diffs.append(
            {
                "key": key,
                "before": before,
                "after": after,
                "delta": delta,
                "judgement": judgement,
                "regression": regression,
            }
        )
    diffs.sort(key=lambda d: (_is_ms(d["key"]) is False, d["key"]))
    return diffs


def has_regression(diffs: list[dict]) -> bool:
    return any(d.get("regression") for d in diffs)


def _config_changes(base: dict, current: dict) -> list[str]:
    b = base.get("config") or {}
    c = current.get("config") or {}
    out = []
    for key in sorted(set(b) | set(c)):
        if b.get(key) != c.get(key):
            before = b.get(key, "（未记录）")
            after = c.get(key, "（未记录）")
            out.append(f"{key} {before} → {after}")
    return out


def _corpus_line(base: dict, current: dict) -> str:
    b = base.get("corpus") or {}
    c = current.get("corpus") or {}
    if b == c:
        return f"语料：{c.get('documents')} 篇 / {c.get('chunks')} 切片（未变）"
    return (
        f"语料：{b.get('documents')} 篇 / {b.get('chunks')} 片 → "
        f"{c.get('documents')} 篇 / {c.get('chunks')} 片  ⚠ 语料变了，指标不可直接横比"
    )


def format_diff(diffs: list[dict], base: dict, current: dict, show_all: bool = False) -> str:
    total = len(current.get("metrics") or {})
    lines = [""]
    lines.append(
        f"对比 baseline：{base.get('generated_at', '?')}（指纹 {base.get('config_hash')}）"
        f" → {current.get('generated_at', '?')}（指纹 {current.get('config_hash')}）"
    )
    changes = _config_changes(base, current)
    if changes:
        lines.append("配置变化：" + "；".join(changes))
        if base.get("config_hash") != current.get("config_hash"):
            lines.append("  ⚠ 指纹不同 = 不是同一组实验。数字可比，但结论要分开讲。")
    else:
        lines.append("配置：未变")
    lines.append(_corpus_line(base, current))
    for label, payload in (("基线", base), ("本轮", current)):
        for note in payload.get("caveats") or []:
            lines.append(f"⚠ {label}保留说明：{note}")

    if not diffs:
        lines.append("")
        lines.append(f"指标：{total} 项全部未变。改动没生效，或者改动跟指标无关——两种情况都值得查。")
        return "\n".join(lines) + "\n"

    shown = diffs if show_all else diffs
    lines.append("")
    lines.append(f"指标变动 {len(shown)} 项（共 {total} 项，{total - len(diffs)} 项未变，未变的默认不列）")
    head = ("指标", "基线", "本轮", "变化", "判定")
    lines.append(f"{head[0]:<34}{head[1]:>12}{head[2]:>12}{head[3]:>12}  {head[4]}")
    lines.append("-" * 82)
    for item in shown:
        key = item["key"]
        delta = item["delta"]
        lines.append(
            f"{key:<34}{fmt(key, item['before']):>12}{fmt(key, item['after']):>12}"
            f"{(fmt_delta(key, delta) if delta is not None else '—'):>12}  {item['judgement']}"
        )
    lines.append("-" * 82)
    bad = [d["key"] for d in diffs if d.get("regression")]
    if bad:
        lines.append(f"倒退 {len(bad)} 项：{'、'.join(bad)}")
    else:
        lines.append("没有倒退项。")
    return "\n".join(lines) + "\n"


def load(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"找不到结果文件：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    here = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="对比两份评测结果")
    parser.add_argument("--baseline", default=str(here / "eval" / "baseline.json"))
    parser.add_argument("--current", default=str(here / "eval" / "results" / "latest.json"))
    parser.add_argument("--all", action="store_true", help="连未变的指标一起列")
    parser.add_argument("--fail-on-regression", action="store_true", help="有倒退时退出码 1")
    args = parser.parse_args()

    base = load(Path(args.baseline))
    current = load(Path(args.current))
    diffs = diff_payloads(base, current)

    if args.all:
        base_metrics = base.get("metrics") or {}
        cur_metrics = current.get("metrics") or {}
        listed = {d["key"] for d in diffs}
        for key in cur_metrics:
            if key not in listed and key in base_metrics:
                diffs.append(
                    {
                        "key": key,
                        "before": base_metrics[key],
                        "after": cur_metrics[key],
                        "delta": 0.0,
                        "judgement": "未变",
                        "regression": False,
                    }
                )

    print(format_diff(diffs, base, current, show_all=args.all))
    if args.fail_on_regression and has_regression(diffs):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
