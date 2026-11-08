#!/usr/bin/env python3
"""重建 BM25 索引并落盘 pkl（离线运维用；正常运行由入库流程自动触发）。

什么时候需要手动跑
----------------
- 换过 `CHUNK_SIZE` / `CHUNK_OVERLAP` 之后重建过向量库：两套索引的切片口径必须一致
- 改过分词口径（`app/tokenize.py` 的 TOKENIZER_VERSION）或 BM25 参数
- 从别处拷来一份 chroma_db 但没带 pkl

用法
----
    python scripts/build_bm25.py            # 重建 + 落盘
    python scripts/build_bm25.py --check    # 只校验磁盘上的索引是否与语料一致
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.bm25_index import BM25Index, fingerprint  # noqa: E402
from app.config import get_settings  # noqa: E402
from eval.eval import load_chunks  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="重建 BM25 索引")
    parser.add_argument("--chroma-dir", default=None, help="覆盖向量库路径")
    parser.add_argument("--check", action="store_true", help="只校验，不重建")
    args = parser.parse_args()

    settings = get_settings()
    rows_all = load_chunks(args.chroma_dir or settings.chroma_dir, settings.collection_name)
    if not rows_all:
        print("向量库是空的，先入库文档再来。")
        return 1

    docs = len({str(meta.get("doc_id") or "") for meta, _ in rows_all})
    print(f"语料：{docs} 篇 / {len(rows_all)} 切片 | 指纹 {fingerprint([(str(m.get('doc_id'))+':'+str(m.get('chunk_index')), m, t) for m, t in rows_all])}")

    index = BM25Index(settings)
    if args.check:
        payload = index._read_pickle()
        if payload is None:
            print("磁盘上没有可用的 pkl（或版本/参数已过期）。")
            return 1
        same = payload.get("fingerprint") == fingerprint(
            [(f"{m.get('doc_id')}:{int(m.get('chunk_index') or 0)}", m, t) for m, t in rows_all]
        )
        print("索引与语料一致" if same else "索引已过期，需要重建")
        return 0 if same else 1

    # 复用 VectorStore 的 dump_all 口径（key = doc_id:chunk_index），避免两处各拼一次 key
    from app.llm_factory import build_embeddings
    from app.vector_store import VectorStore

    store = VectorStore(settings, build_embeddings(settings))
    index.rebuild(store.dump_all(), reason="手动重建")
    print(f"完成：{index.size} 切片 → {settings.bm25_index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
