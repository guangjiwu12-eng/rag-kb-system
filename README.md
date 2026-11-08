# RAG 知识库问答系统 · 消防规范

给**消防规范**做的私有问答服务：14 部国标/行标、4106 个按条切片，提问后由大模型**只依据检索到的原文**作答，并在句末标注 `[n]`；每条引用可回溯到「哪本规范、第几页、第几条原文」。语料里没有的内容直接拒答，不编。

面向的场景不是闲聊，是**查条文**：消防设计、验收、维保里那种「拍脑袋记不住、翻 PDF 翻半天、又不敢信通用 AI」的问题。

## 现在的水平

| 指标 | 数值 | 说明 |
| --- | ---: | --- |
| 严格 `hit@6` | **97.5%** | 只认主承载文档，40 道有答案题 |
| 宽松 `hit@20` | **100%** | 召回已饱和，剩下全是排序问题 |
| 端到端作答率 | **95.0%** | 检索命中且模型确实给出答案 |
| 真幻觉率 | **0%**（0/10） | 语料外提问硬答的比例；样本小，只说明这批题没漏 |
| 检索延迟 p50 / p95 | **95 / 109 ms** | 含 Embedding 与 BM25 融合，不含生成 |

配置指纹 `bf6830`（`kb_v2` + `hybrid_rrf` + `top_k=6`），可一条命令复现，见「评测」。

## 功能

- **文档接入**：PDF（逐页解析，物理页与印刷页两个页码都存）、TXT（UTF-8 / GB18030 自动识别），单文件 20 MB 上限
- **按条切片**：认得出条文号就按「第 X.X.X 条」切（`chunk_mode=auto`），认不出退回 800 字定长；**条文正文与条文说明分离**并打标签；跨标准完全重复的切片合并
- **六档检索**：`vector` / `bm25` / `hybrid_sum` / `hybrid_rrf`（默认）/ `vector_rerank` / `hybrid_rrf_rerank`，可按请求体 `mode` 覆盖，扫矩阵不用重启
- **中文分词**：`lcut_for_search` + 型号保护（`GB50974`、`IP55` 不被切碎）+ 短停用词表
- **引用溯源**：回答中的 `[1][2]` 与接口返回的 `citations` 一一对应；每条引用包含文件名、页码、段号、排序分数、命中原文，以及「是否被答案真正引用」标记
- **拒答**：查询级闸门（`MIN_SCORE`）+ Prompt 判据两道，语料外的问题不调模型或明确说「资料里没有」
- **文档管理**：列表、删除（连同 `data/` 下的原始文件一起删）；同一文件重复上传是幂等的，内容完全一致时连 Embedding 都不重算
- **可评测**：50 题基线 + 12 道改写题 + 10 题 smoke 闸门 + 六档消融矩阵 + 引用自校验
- **安全**：可选的 `X-API-Key` 鉴权、CORS 默认只放行本机来源，避免把私有库暴露在局域网
- **工程化**：配置集中管理、业务异常统一错误码、请求耗时头、健康检查、`POST /admin/reload` 免重启换库、Docker 非 root 运行、99 个不联网的单元/接口测试

## 请求流程

```
上传：PDF/TXT → 逐页解析 → 清洗 → 判文体(够不够条文化?) → 按条切/定长切 → 去重 → Embedding → Chroma
                                    └→ 条文 / 条文说明分离并打标签；同步重建 BM25 索引

提问：问题 ─┬→ bge-m3 向量召回 top20 ─┐
           └→ BM25 关键词召回 top20 ─┴→ RRF(k=60) 融合 → [可选重排] → 取 top6
                                                              ↓
                                       查询级闸门：top1 < MIN_SCORE → 拒答（不调模型）
                                                              ↓
                                   Prompt（只依据原文 + 句末标号）→ LLM → 解析 [n]
                                                              ↓
                    answer + citations(file / page / page_printed / chunk / score / 原文 / used)
```

## 目录结构

```
rag-knowledge-base/
├── app/
│   ├── config.py            # pydantic-settings 配置（.env → Settings）
│   ├── schemas.py           # 接口出入参（含 mode / rerank_budget_ms 两个消融入口）
│   ├── errors.py            # 业务异常（状态码 + 稳定错误码）
│   ├── corpus.py            # 条文化判定、按条切片、条文说明分离、去重
│   ├── document_loader.py   # PDF/TXT 解析、清洗、页码对齐、文件名安全化
│   ├── tokenize.py          # 中文分词 + 型号保护 + 停用词
│   ├── vector_store.py      # Chroma 封装：余弦检索、按文档幂等写入、文档聚合
│   ├── bm25_index.py        # BM25 索引：构建 / 落盘 / 加载 / 检索
│   ├── reranker.py          # bge-reranker-v2-m3：torch / onnx 双后端、预算闸门
│   ├── retrieval.py         # 六档检索编排、融合、深度对齐
│   ├── llm_factory.py       # 模型工厂：云端 / 本地 Embedding、DeepSeek / 通义千问
│   ├── local_embeddings.py  # 本地 bge-m3：CLS 池化 + L2 归一化（惰性加载）
│   ├── rag_service.py       # RAG 核心：入库、检索、Prompt、引用映射、拒答
│   ├── logging_conf.py      # 日志格式
│   ├── main.py              # FastAPI 入口、路由、异常处理、静态页
│   └── static/index.html    # 免构建的单页前端
├── tests/                   # 99 个用例：切片、分词、融合、拒答、接口、错误分支
├── eval/                    # 评测
│   ├── questions.yaml             # 50 题（40 有答案 + 10 无答案）
│   ├── questions_rewritten.yaml   # 12 道改写题（同一事实、日常问法）
│   ├── smoke_questions.yaml       # 10 题冒烟集，门槛在 smoke_floors.json
│   ├── ablation.py                # 六档检索矩阵
│   ├── eval.py                    # 双路评测 + 阈值 sweep + 定基线
│   └── baseline.json              # 当前基准（配置指纹 bf6830）
├── scripts/                 # 语料与诊断工具链
│   ├── ingest_dir.py        # 批量入库 + 切片体检（片/页 一栏定生死）
│   ├── rebuild_index.py     # 全量重建（默认只出报告，--apply 才写库）
│   ├── build_bm25.py        # 给「拷来的库」补 BM25 索引
│   ├── ocr_pdf.py           # 扫描件 → 带文字层 PDF（保留原页码）
│   ├── build_corpus_pdf.py  # 纯文本 → 分页 PDF（网页语料入库用）
│   ├── verify_citations.py  # 引用自校验：答案里的数值是否真在原文
│   ├── diagnose_refusals.py # 逐题摆出「命中却拒答」的原文与回答
│   ├── find_carriers.py     # 反查某句话在库里由哪些切片承载
│   ├── diff_eval.py         # 与 baseline 对比，只报变动的指标
│   ├── smoke_gate.py        # 10 题冒烟闸门，跌破地板值即失败
│   ├── bench_rerank.py      # 重排单段压测：窗口 × 候选数 × 后端
│   └── export_reranker_onnx.py   # 重排模型导 int8 ONNX（2.27 GB → 568 MB）
├── docs/03-retrieval.md     # 检索设计：六个开关、四个决策、重排为什么慢
├── models/                  # 本地权重（bge-m3 2.3 GB、reranker 2.27 GB，已 gitignore）
├── data/                    # 原始文件：inbox/ 待入库、sources/ 语料正文、bm25_index.pkl
├── chroma_v2/               # 向量库持久化（kb_v2 集合，4106 片）
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-embedding.txt   # 可选：本地 Embedding 的 torch / transformers
├── requirements-rerank.txt      # 可选：重排档才需要
└── .env.example
```

## 快速开始

### 1. 密钥与权重

```bash
cp .env.example .env
# DEEPSEEK_API_KEY：问答；DASHSCOPE_API_KEY 仅在 EMBEDDING_BACKEND=dashscope 时需要
```

本项目走本地模型（不花钱、不出网），权重从 ModelScope 拉（HuggingFace 国内常拉不动）：

```bash
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('BAAI/bge-m3', local_dir='models/bge-m3')"
python -c "from modelscope import snapshot_download; snapshot_download('BAAI/bge-reranker-v2-m3', local_dir='models/bge-reranker-v2-m3')"
```

### 2. 本地运行

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements-dev.txt          # 含 torch，约 2 GB
pip install -r requirements-rerank.txt       # 可选：要跑重排档才需要
uvicorn app.main:app --port 8000
```

- 对话页面：<http://127.0.0.1:8000/>
- 接口文档：<http://127.0.0.1:8000/docs>
- 健康检查：<http://127.0.0.1:8000/api/v1/health>
- 检索链路状态：<http://127.0.0.1:8000/api/v1/retrieval>（一眼看 BM25 是否与向量库同步、重排走 torch 还是 onnx）

**两个必踩的坑：**

1. **`http_proxy` 会把本机请求劫持走**，表现成连接超时/被拒。起服务前清掉：
   `export NO_PROXY='*' http_proxy= https_proxy=`。
2. **`EMBEDDING_MODEL` 必须是「跑代码的那个 python」看得见的路径**（见下文「三个不显眼的坑」第 1 条）。

### 3. Docker

```bash
mkdir -p data chroma_db        # 首次运行前创建，避免被 Docker 建成 root 目录
docker compose up --build -d
```

停止：`docker compose down`

> compose 里挂的是 `chroma_db` + 默认集合，模型走云端 Embedding。要用本机的 `chroma_v2` + 本地 bge-m3，就别用 Docker，直接按上一节本地跑（`CHROMA_DIR` 指到该目录，集合 `kb_v2`）。

### 4. 换成本地 Embedding（bge-m3，不花钱、不出网）

默认走通义千问 `text-embedding-v1`（1536 维）。想完全离线、也不再消耗 Embedding 额度，换成本地 bge-m3：

```bash
pip install -r requirements-embedding.txt          # torch + transformers，约 2 GB
# 权重：HuggingFace 在国内常拉不动，走 ModelScope
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('BAAI/bge-m3', local_dir='models/bge-m3')"
```

`.env` 里改四处（`models/` 已在 `.gitignore`，权重不进仓库）：

```bash
EMBEDDING_BACKEND=sentence-transformers
EMBEDDING_MODEL=/绝对路径/models/bge-m3            # 必须绝对路径
EMBEDDING_NUM_THREADS=16                          # CPU 上影响极大，见下表
EMBEDDING_BATCH_SIZE=16
```

⚠️ **维度从 1536 变 1024，必须清空 `chroma_db/` 重新入库**——Chroma 不接受同一集合混两种维度。

本机实测（12 线程 CPU，1335 片）：线程数是唯一值得调的参数，其余保持默认。

| 线程 | 吞吐 | 1335 片耗时（约 104 万字） |
| --- | --- | --- |
| 12 | 259 字/秒 | 约 65 分钟 |
| 16 | 579 字/秒 | 约 30 分钟（本轮端到端 41 分钟，含 PDF 解析与写库） |

代价是没有 batch 化调优空间，改一次切片就要重算全库；好处是零 API 成本、数据不出机器，
且中文短句检索比 `text-embedding-v1` 强不少（60 字以内的条款问句差距最明显）。

#### 三个不显眼的坑（都会表现成「卡住」，不报错）

1. **`EMBEDDING_MODEL` 必须对写代码的那个 python 成立。** 用 WSL 里的 python 跑就写
   `/home/xxx/models/bge-m3`；用 Windows 侧的 python 跑同一个仓库（`//wsl$/...`）就写相对项目根的
   `models/bge-m3`——那个 `/home/...` 在 Windows 侧不存在。路径不存在时 transformers 不会报错，
   它会当成 HF 仓库名去联网下载，网络不通就卡在重试上。现在这类路径会在加载前直接抛错
   （`app/local_embeddings.py` 的 `_resolve_source`）。
2. **加 `HF_HUB_OFFLINE=1`。** 本地已经有权重了，别给任何「顺便联网」的机会。
3. **Chroma 的库目录别放在 `\\wsl$` 这类 UNC 路径上。** 实测 `PersistentClient` 在 UNC 上初始化
   会卡住（同一操作在本地盘 0.1 秒）；`chroma_db/` 在 WSL 里、python 在 Windows 上时尤其容易踩。
   要么两边都用 WSL，要么把 `CHROMA_DIR` 指到本地盘（如 `D:/workbuddy-space/chroma_kb2`，
   WSL 侧照常能通过 `/mnt/d/...` 读到）。

遥测默认是关的（`app/vector_store.py` 里显式传 `anonymized_telemetry=False`）：私有文档库不该
往外发数据，而且没有外网的机器上，遥测上报会在每次写入时阻塞到超时。

## 检索档位

六个开关都在 `RETRIEVAL_MODE`，也可以在 `/search`、`/query` 的请求体里用 `mode` 临时覆盖——消融矩阵不用改配置重启服务：

| 档位 | 做什么 | 什么时候用 |
| --- | --- | --- |
| `vector` | bge-m3 余弦 top20 | 基线对照 |
| `bm25` | 只用稀疏检索 | 诊断「是不是词面重合在起作用」 |
| `hybrid_sum` | 两路归一化后加权（0.5/0.5） | 对照融合方式 |
| `hybrid_rrf` | RRF(k=60) 按名次融合 | **默认** |
| `vector_rerank` | 向量 top20 → cross-encoder 重排 | 质量上限，延迟不可接受 |
| `hybrid_rrf_rerank` | RRF top20 → 重排 | 同上 |

40 道有答案题，`k=6`，召回深度 20，重排闸门关闭（= 质量上限，`eval/results/kb_v2/ablation-*.md`）：

| 档位 | hit@6（严） | hit@6（宽） | MRR | p50 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 纯向量 | 95.0% | 97.5% | 0.7751 | 91 ms | 105 ms |
| +BM25（加权和） | 97.5% | 97.5% | 0.8353 | 114 ms | 132 ms |
| **+RRF（默认）** | **97.5%** | **97.5%** | **0.8531** | 123 ms | 141 ms |
| 纯 BM25（对照） | 97.5% | 97.5% | 0.8438 | 9 ms | 14 ms |
| +rerank | 100% | 100% | 0.9458 | 9807 ms | 11497 ms |
| +RRF+rerank | 100% | 100% | 0.9208 | 8734 ms | 11456 ms |

三条结论必须一起说：

1. **重排是质量上限，不是默认档**。它把 hit@6 拉满、MRR 推到 0.9458，代价是 p95 从 141ms 涨到 11.5s（约 80 倍）。交互式问答里这个交易不成立；线上 400ms 预算下重排覆盖率为 0（闸门整档跳过）。真解法是「先抽与问题最相关的句子再重排」，成本能降一个量级。
2. **BM25 的边际收益只有 +2.5pp hit@6**（MRR +0.078）。旧语料上它是 +17.5pp 的主力——按条切片已经把「答案在长文档里被稀释」这个洞补上了，而 BM25 原本补的就是这个洞。**两轮收益不可相加。**
3. **「纯 BM25 97.5%」是题库造成的假象**。50 题里问法与规范用词高度重合，词面重合就能命中；换 12 道改写题后纯 BM25 掉到 66.7%，向量与融合档仍是 91.7%（详见「评测」）。

## API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/api/v1/documents` | 上传并索引文档（multipart，字段名 `file`），201 |
| `GET` | `/api/v1/documents` | 文档列表（按入库时间倒序，含切片数、页数、大小） |
| `DELETE` | `/api/v1/documents/{doc_id}` | 删除文档、全部切片，以及 `data/` 下的原始文件 |
| `POST` | `/api/v1/search` | **只检索不生成**：返回深度内的原始命中，不过 `MIN_SCORE`、不调大模型（评测/调参用） |
| `POST` | `/api/v1/query` | 提问，返回回答与引用 |
| `GET` | `/api/v1/health` | 健康检查：模型配置、集合、文档/切片数、BM25 片数（无需鉴权） |
| `GET` | `/api/v1/retrieval` | 检索链路状态：当前档位、BM25 是否与向量库同步、重排后端 |
| `POST` | `/api/v1/admin/reload` | 免重启重载索引，返回重载前后计数；`changed=false` 说明库本来没变 |

除健康检查外，所有接口在设置了 `API_KEY` 之后必须携带请求头 `X-API-Key`（未设置则保持开放，方便本地开发）。
示例统一缩写为 `H='X-API-Key: your-key'`：

```bash
curl -H "$H" ...
```

### 上传

```bash
curl -X POST http://127.0.0.1:8000/api/v1/documents \
  -H "X-API-Key: your-key" \
  -F "file=@GB50974-2014_消防给水及消火栓系统技术规范.pdf"
```

```json
{
  "doc_id": "7b93949de94cae86",
  "file": "GB50974-2014_消防给水及消火栓系统技术规范.pdf",
  "file_type": "pdf",
  "pages": 320,
  "chunks": 812,
  "replaced": false,
  "elapsed_ms": 8421
}
```

### 仅检索

```bash
curl -X POST http://127.0.0.1:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"question": "排烟风机应能在多少度烟气下连续运行 30 分钟", "top_k": 20, "mode": "hybrid_rrf"}'
```

```json
{
  "question": "排烟风机应能在多少度烟气下连续运行 30 分钟",
  "hits": [
    {
      "index": 1,
      "doc_id": "a1f3c2e58b7d4901",
      "file": "GB51251-2017_建筑防烟排烟系统技术标准.pdf",
      "page": 33,
      "page_printed": 27,
      "chunk_index": 148,
      "score": 0.0313,
      "content": "4.4.5 排烟风机应能在 280℃ 的环境下连续运行 30min……",
      "used": false
    }
  ],
  "mode": "hybrid_rrf",
  "elapsed_ms": 118
}
```

`hits` 按分数降序，**不做阈值过滤**，`used` 恒为 `false`（没有答案可引用）。
是否超过 `MIN_SCORE` 由调用方自己比——这正是评测与调阈值需要的原始信号。

### 提问

```bash
curl -X POST http://127.0.0.1:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"question": "消防水泵房设在地下室时有什么要求？"}'
```

```json
{
  "answer": "消防水泵房设在地下室（附设在建筑物内）时：不应设置在地下三层及以下，或室内地面与室外出入口地坪高差大于 10m 的地下楼层 [2]；应采用耐火极限不低于 2.0h 的隔墙和 1.50h 的楼板与其他部位隔开 [2]；疏散门应直通安全出口 [2]。条文说明补充：出口宜直通安全出口，泵房应设放水阀和排水措施 [1][3]。",
  "citations": [
    {
      "index": 2,
      "doc_id": "7b93949de94cae86",
      "file": "GB50974-2014_消防给水及消火栓系统技术规范.pdf",
      "page": 54,
      "page_printed": 54,
      "chunk_index": 202,
      "score": 0.0325,
      "content": "5.5.12 消防水泵房……",
      "used": true
    }
  ],
  "model": "deepseek-chat",
  "mode": "hybrid_rrf",
  "elapsed_ms": 4210,
  "abstained": false
}
```

几个字段要看清：

- **`score` 是「决定排序的那个分」，不是余弦相似度。** 融合档里它是 RRF 分（量级 `1/(60+rank)` ≈ 0.016），拿它跟 `MIN_SCORE=0.30` 逐片比会把所有命中砍光——所以拒答判断用的是**查询级闸门**（只比 top1），逐片滤噪只在 `vector` 档保留。这个坑在合并两版代码时踩过一次，`tests/test_retrieval_merge.py` 钉着。
- `page` 对 TXT 为 `null`；`page_printed` 是页脚印的印刷页，可能与物理页差一个常数。
- `used` 表示回答正文里是否出现了对应的 `[n]`。
- `abstained=true` 表示这次是拒答（top1 未过闸门，未调用大模型）。

### 错误格式

所有业务错误统一为：

```json
{ "detail": "仅支持 PDF / TXT 文件，当前为：.docx", "code": "unsupported_file_type" }
```

常见错误码：`401 unauthorized`（缺/错 `X-API-Key`）、`415 unsupported_file_type`、`413 file_too_large`、`422 no_text_extracted`（扫描版 PDF）、`404 document_not_found`、`503 model_not_configured`（缺 Key）、`502 model_invocation_failed`。

## 配置项

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DASHSCOPE_API_KEY` | — | `EMBEDDING_BACKEND=dashscope` 时必填；走本地模型可以不填 |
| `DEEPSEEK_API_KEY` | — | `LLM_PROVIDER=deepseek` 时必填 |
| `LLM_PROVIDER` | `deepseek` | `deepseek` / `dashscope` |
| `LLM_MODEL` | `deepseek-chat` | 切换 provider 时会自动兜底为 `qwen-plus` |
| `EMBEDDING_BACKEND` | `dashscope` | `dashscope` / `sentence-transformers`（本地） |
| `EMBEDDING_MODEL` | `text-embedding-v1` | 本地时填权重目录绝对路径。**换模型必须清空向量库**，维度不同 |
| `EMBEDDING_DEVICE` | `cpu` | 本地模型的设备：`cpu` / `cuda` |
| `EMBEDDING_BATCH_SIZE` | `16` | 本地模型的批大小 |
| `EMBEDDING_NUM_THREADS` | 空 | torch 线程数；CPU 上影响很大（本机 12→16 线程吞吐 ×2.2） |
| `EMBEDDING_NORMALIZE` | `true` | 向量归一化后再算余弦，bge 系列按此训练 |
| `EMBEDDING_MAX_SEQ_LENGTH` | 空（代码内 1024） | 本地模型的最大 token 数 |
| `CHUNK_MODE` | `auto` | `auto` / `article` / `fixed`；认不出条文号自动退回定长 |
| `ARTICLE_MIN_MARKERS` / `ARTICLE_MIN_DENSITY` | `12` / `0.06` | 判定「这是规范」的双条件：条文号个数 **且** 行占比 |
| `ARTICLE_MAX_CHARS` / `ARTICLE_SUB_OVERLAP` | `1200` / `80` | 单条过长才二次切分 |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `150` | 仅定长口径生效 |
| `DEDUPE_ENABLED` / `DEDUPE_THRESHOLD` | `true` / `0.90` | 完全一致的才合并；近重复只报告不合并 |
| `RETRIEVAL_MODE` | `hybrid_rrf` | 六档之一，见「检索档位」 |
| `RECALL_CANDIDATES` | `20` | 每一路的召回深度，也是最终保留深度 |
| `RRF_K` | `60` | 论文默认值，本题库未扫过 |
| `BM25_K1` / `BM25_B` / `BM25_WEIGHT` | `1.5` / `0.75` / `0.5` | 后者只作用于 `hybrid_sum` |
| `BM25_INDEX_PATH` | `./data/bm25_index.pkl` | 入库时重建，重启直接读，不重新分词 |
| `RERANK_MODEL` / `RERANK_ONNX_PATH` | `models/bge-reranker-v2-m3` / 空 | ONNX 目录存在时优先走 onnxruntime |
| `RERANK_MAX_LENGTH` | `640` | 按切片长度分布定（中位 502 / p90 574），256 会让重排看不到答案 |
| `RERANK_BUDGET_MS` | `0` | 0 = 不设预算；非 0 时超预算整档跳过重排 |
| `RERANK_MIN_CANDIDATES` / `RERANK_FAIL_OPEN` | `5` / `true` | 候选太少或模型报错时降级为融合顺序 |
| `TOP_K` | `6` | 每次检索喂给模型的片段数（可由请求体 `top_k` 覆盖）。`k=4` 时代 n11 的正确答案稳定排第 5，纯截断 |
| `MIN_SCORE` | `0.30` | **查询级**闸门：top1 低于它直接拒答、不调模型；调到 0 等于关闭。**本项目用三档双路评测定为 `0.30`**，不是 sweep 的 F1 最优点，理由见「评测」 |
| `API_KEY` | 空 | 设置后 `/api/v1/*`（除健康检查）要求 `X-API-Key` 请求头 |
| `ALLOWED_ORIGINS` | `http://localhost:8000,http://127.0.0.1:8000` | CORS 白名单，逗号分隔 |
| `MAX_UPLOAD_MB` | `20` | 上传大小上限 |
| `DATA_DIR` / `CHROMA_DIR` | `./data` / `./chroma_db` | 原始文件与向量库目录；本项目实际用 `D:/workbuddy-space/chroma_kb2` |
| `COLLECTION_NAME` | `kb_documents` | Chroma 集合名；本项目实际用 `kb_v2` |

## 设计说明

- **为什么用余弦空间**：Chroma 默认 L2，文本 Embedding 通常用余弦更稳定；建集合时指定 `{"hnsw:space": "cosine"}`，接口把距离换算成 `1 - distance` 的相似度。
- **为什么 RRF 而不是加权分数**：两路分数量纲不同（余弦 0~1，BM25 无上界），归一化会引入新假设；RRF 只用名次，不需要标定。代价是丢掉「第一名的领先幅度」这个信号，这也是它 MRR 不如重排档的原因。
- **深度对齐**：所有档位最终都吐 20 条候选、取 top6。不比「谁拿到的候选更多」，只比「融合与重排」本身，消融矩阵才有意义。
- **幂等入库**：`doc_id = sha256(文件内容)[:16]`，切片 ID 固定为 `doc_id:段号`；按 `doc_id` 或同名文件找出旧切片，**写入成功后才删除**，所以重复上传不会产生脏数据，改名覆盖也会清理干净。内容完全一致时连 Embedding 都不重算。
- **先写后删**：唯一的远程调用（Embedding）失败时，旧切片还在库里，文档不会凭空消失；只有同一 `doc_id` 但切片口径变了（改了 `CHUNK_SIZE`）才会退化成先删冲突 ID 再写。
- **宁可拒答，但闸门放在查询级**：向量检索永远会返回 Top-K，哪怕全是噪声，所以要有第二道闸门。它比的是**查询 top1 分**，不是逐片比——融合档的分数是 RRF 分（≈0.016），逐片拿 `0.30` 去比会把所有命中砍光。阈值不是拍脑袋定的：`--sweep` 在 50 题上扫出 F1 最优点 `0.65`，但把三档跑完整双路评测会发现抬阈值只是「少调几次模型」，误拒多 4 道、幻觉一例没少——所以取 `0.30`，只让它挡噪声（三档对照表见「评测」）。
- **标签只摆事实，判据才下结论**：条文说明只标 `【条文说明】`，不写「非规范要求」。早期版本把价值判断塞进标签，模型就把它当禁令，答案只在说明里的题一律拒答（详见「语料」）。
- **引用可信**：Prompt 强制要求「只依据资料 + 句末标注编号 + 资料不足就说不知道」；服务端解析回答中的 `[n]` 回填 `used`，前端对未被引用的片段做弱化展示；另有 `verify_citations.py` 独立核对数值出处。
- **删除干净**：删文档时连同 `data/` 下的原始文件一起删；路径做过越界校验，不会因为元数据里的文件名异常而删到别的地方。
- **默认只对本机开放**：CORS 放行 localhost + 127.0.0.1；配置 `API_KEY` 后写/检索接口强制校验 `X-API-Key`（健康检查保持开放给 Docker HEALTHCHECK）。
- **失败可控**：模型未配置返回 503 而不是 500；扫描版 PDF 返回 422 并提示 OCR；超大文件边读边计数，避免打爆内存；重排报错默认 fail-open 降级为融合顺序，检索可用性优先于排序质量。
- **可测试性**：`RAGService` 支持注入 `Embeddings` / `ChatModel`，测试用假模型即可覆盖完整链路，不联网、不消耗 token。

## 安全约定

- `.env` 永远不进仓库（`.gitignore` + pre-commit 双保险）；`.env.example` 里只放占位符
- 仓库内 `.githooks/pre-commit` 会拦截 `.env`、`*.sqlite3`、`chroma_db/*`、`data/*.pdf` 以及任何长得像密钥的字符串（`-E '(sk-|AKID|...)[A-Za-z0-9_./+=-]{30,}'`），误报可用 `git commit --no-verify` 临时跳过
- 首次克隆后启用闸门：

```bash
git config core.hooksPath .githooks
chmod +x .githooks/pre-commit
chmod 600 .env          # 只让自己可读
```

## 语料

13 份现行消防标准 + 1 份实践日志。同一份语料按两种口径切：

| 来源 | 篇数 | 说明 |
| --- | --- | --- |
| `data/sources/*.txt` | 13 | 标准全文（正文 + 条文说明），抓自建研院防火所公开全文页 |
| `data/inbox/*.pdf` | 13 | 由上面的 txt 用 `scripts/build_corpus_pdf.py` 排成 PDF 后入库 |
| 实践日志 | 1 | 学生实践报告，第一人称叙述，是本库唯一的非规范文体 |

| 口径 | 切片数 | 用在哪 |
| --- | --- | --- |
| 定长 800/150（`chunk_mode=fixed`） | 1335 | 兜底：认不出条文号的非规范文体 |
| 按「第 X.X.X 条」切（`chunk_mode=article`） | 4119 → 去重 4106 | 规范类语料的默认口径 |

### 切片口径

**为什么不用 800 字定长。** 定长切片是按字数走的，不管条文边界。实测三道路由失败的题
（n2、n16、n21），答案句都落在切片尾部——n2 的答案句在 660 字片段的第 589 字（89% 位置），
前 589 字全是「见表 C.1」这类同构噪声，向量被噪声拉走，检索自然排不上来。按条切之后
同一道题的答案句落在 3% 位置，片长 660 → 181 字。

判据不是「找条文号」而是「找得够不够密」：行首条文号 ≥12 个**且**占全部行 ≥6%。
只看绝对数会把实践日志误判成规范——日志里「1.21 地点……」这种日期行也长得像条文号，
13 个就能过阈值；规范的条文号密度在 10%~20%，日志只有 1.9%，密度这一刀把它挡回去了。

**条文 / 条文说明分离。** 这套语料里「同一事实存两份」的真实形态不是两份文件，
而是**正文与逐章条文说明交错排布**。判定靠编号复用：同一编号第二次出现、且文本相似度
<0.90 → `explanation`（条文说明）；≥0.90 → `duplicate`（同号同文，直接丢）。
比找「条文说明」四个字稳，因为那四个字同时也是页眉和导航块。切出来是
正文 2807 / 条文说明 1266 / 前置 33。喂给模型时条文说明带 `【条文说明】` 标签。

标签只标性质，不替模型下结论。早期版本写的是 `【条文说明，非规范要求】`，
结果 n10 / n23 的答案只存在于条文说明里时，模型宁可拒答也不引用——
命中仍拒答 15%、端到端只有 82.5%。去掉「非规范要求」这个判断词、并在 Prompt 里写明
「正文没有规定时条文说明是有效依据」之后，这两道都能答，端到端到 95.0%。
**教训：给模型的标签只负责摆事实，价值判断交给 Prompt，否则它会把提示当成禁令。**

**页码两个口径都存。** `page` 是 PDF 物理页，`page_printed` 是页脚印的印刷页。
偏移量只有整篇一致时才拿来推算，不一致就记 0（0 = 无页码）——宁缺不给错，
猜错的页码比没有页码更费人。`gold` 标注两个口径都认，不用为了对齐重标。

**去重只并完全一致。** 比对前先剥掉开头的条文号，否则同一段文字换个标准换个编号就认不出来。
剥掉之后找到 13 组跨标准完全重复（GB50166 6.0.4 ≡ GB51309 7.0.4、GB50974 10.3.1 ≡
GB50084 9.3.1 等），正本切片的 `also_in` 记着「这段文字还出现在哪」，删除的是重复向量，
不是证据。近似重复（4-gram Jaccard ≥0.90）只报告不合并——规范里 6.2.5~6.2.19 十九条巡查
内容只差系统名，相似度能到 0.8 以上，按相似度合并会把十九条不同的要求压成一条。

全量重建：`python scripts/rebuild_index.py` 只出报告，加 `--apply --collection kb_v2`
才真写库，验收完再改配置切过去。

两件必须知道的事：

1. **页码是排版页，不是原书页码**。`build_corpus_pdf.py` 每页约 740 字、优先在中文标点处折行，
   页码只在本库内部自洽。`questions.yaml` 里 gold 的 `page` 只能对着本库解释，别拿它去查纸质标准。
2. **扫描件没进库**。最初直接从消防总队官网下的 11 份标准 PDF 全是纯图片（PyMuPDF 抽 0 字符、
   每页都是图），按流程该走 OCR：`scripts/ocr_pdf.py` 能把扫描件转成「保留原页码 + 不可见文字层」
   的 PDF（实测 13 页 → 112 KB，pypdf 可正常抽取）。但本机 OCR 约 30 s/页，314 页要两个多小时，
   于是改走「网页全文 → 排版 PDF」这条更快也更干净的路。OCR 脚本保留，换台快机器可直接用。

`data/_excluded/232吴光济实验2.pdf` 只有 8 片、主题与规范不搭，已移出数据目录不入库。

## 评测

`eval/` 是可复跑的评测，打的是**真实接口**：`/search` 测检索，`/query` 测拒答。

```bash
uvicorn app.main:app --port 8000        # 先起服务，评测打的是真实接口

python eval/eval.py             # 双路评测：三套口径 + 拒答分解 + 逐题归类
python eval/eval.py --no-query  # 只测检索层，不花 token（改标注后先跑这个）
python eval/eval.py --sweep     # 扫 MIN_SCORE 0.00→0.90，输出 F1 曲线与建议值
python eval/eval.py --set-baseline   # 把本轮定为基准
python scripts/diff_eval.py --fail-on-regression   # 只报变动的指标，退化时非零退出

# 六档消融矩阵（--base-url 指到你起的服务）
python -m eval.ablation --base-url http://127.0.0.1:8000 --k 6 --depth 20 \
    --with-extra --modes vector,bm25,hybrid_sum,hybrid_rrf --out-dir eval/results/kb_v2

python scripts/smoke_gate.py    # 10 题冒烟，跌破 smoke_floors.json 的地板值即失败
python scripts/verify_citations.py   # 引用自校验：答案里的数值是否真在原文
```

| 指标 | 定义 | 分母 |
| --- | --- | --- |
| `strict_hit@k` | 只认主承载文档（gold 里没标 `alternate` 的锚点） | 有答案且 gold 解析成功的题 |
| `loose_hit@k` | 任一承载文档都算（同一句话写在两部标准里时不判失败） | 同上 |
| 端到端 | 检索命中（宽松）**且**模型确实作答 | 同上 |
| `strict_mrr` / `loose_mrr` | 首个 gold 出现位次的倒数，未命中记 0 | 同上 |
| `answerable_refused_rate` | 有答案题里模型没给出答案的比例 | 全部有答案题 |
| ├ `refused_after_miss` | 检索没排进 top-k → 拒答 | 同上 |
| ├ `refused_despite_hit` | 原文就在 top-k 里 → 仍拒答（Prompt / 长上下文问题） | 同上 |
| └ `answered_after_miss` | 检索没排进 top-k → **照样给答案**（硬答） | 同上 |
| `拒答·阈值路径` | 无答案题被 `MIN_SCORE` 拦掉的比例 | 无答案题 |
| `拒答·模型路径` | 阈值没拦、但模型自己说「无法回答」的比例 | 无答案题 |
| `真幻觉率` | 两条路都没拦住、模型硬答的比例 | 无答案题 |

### 本轮结果（`baseline.json`，指纹 `bf6830`）

库 **14 篇 / 4106 片**，Embedding **bge-m3（本地，1024 维）**，`RETRIEVAL_MODE=hybrid_rrf`，`TOP_K=6`，`MIN_SCORE=0.30`，题库 50 道（40 有答案 + 10 无答案）。逐题数据在 `eval/results/kb_v2/`：

| 指标 | 数值 |
| --- | ---: |
| `strict_hit@6` / `loose_hit@6` | **97.50%** / **97.50%** |
| `strict_mrr` / `loose_mrr` | 0.8313 / 0.8500 |
| `caliber_gap`（严宽差） | 0.00% |
| `e2e_answer_rate`（端到端作答） | **95.00%** |
| `answerable_refused_rate` | 2.50%（1 道，全部是「排到仍拒」） |
| ├ `refused_after_miss` | 0.00% |
| └ `answered_after_miss` | 2.50%（1 道，最危险的一类） |
| `abstain_by_threshold` / `abstain_by_llm` | 0% / 100% |
| `false_answer_rate` | **0.00%**（0/10） |
| 检索延迟 p50 / p95 | **95 / 109 ms** |

拒答分解（分母 40 道有答案题，四支相加 = 40）：

| 分支 | 比例 | 道数 |
| --- | --- | --- |
| 检索命中且作答（= 端到端） | 95.00% | 38 |
| 命中但模型仍拒答 | 2.50% | 1 |
| 未命中 → 拒答 | 0.00% | 0 |
| 未命中 → 仍硬答 | 2.50% | 1 |

**瓶颈已经不在检索侧。** 四档非重排档的 hit@20 全是 100%，召回饱和；端到端与 hit@6 之间那点差额全部来自「排到了却拒答」。这就是最后一步为什么改 Prompt 而不是改检索。

### 两次提升，各自独立

| 步骤 | 改了什么 | 指标变化 |
| --- | --- | --- |
| ① 切片口径 | 800 定长（1335 片）→ 按条切（4106 片），纯向量档 | 严格 hit@6 **82.5% → 95.0%**，MRR 0.5887 → 0.7751 |
| ② 检索融合 | 纯向量 → `hybrid_rrf` | 严格 hit@6 **95.0% → 97.5%**，MRR 0.7751 → 0.8531 |
| ③ 答案侧判据 | 标签放宽 + 判据从「直接写明」到「原文里有这个内容」 | 端到端 **82.5% → 95.0%**，命中仍拒答 15.0% → 2.5%，真幻觉仍 0% |

第 ①、② 步的收益**不可相加**：BM25 在旧语料上是 +17.5pp 的主力，在按条切片后只剩 +2.5pp——它补的洞已经被切片方式补掉了。第 ③ 步是纯答案侧改动，检索指标一字未动，正好证明瓶颈位置。

### 「命中却拒答」的逐题诊断

端到端一度卡在 82.5%，而 hit@6 已经 97.5%——差的 15pp 全部在答案侧。用 `scripts/diagnose_refusals.py` 逐题摆出「命中的原文」和「模型的回答」后，归为三类：

| 题 | 现象 | 性质 | 处置 |
| --- | --- | --- | --- |
| n10、n23 | 答案只写在**条文说明**里，模型看到 `【条文说明，非规范要求】` 标签后拒绝据此作答 | 标签把「解释」变成了禁区：能分清要求与解释，也把只有解释承载的题判成了无据 | 标签改成 `【条文说明】` 只标性质；Prompt 写明正文没有规定时条文说明是有效依据 |
| q007、n20、n29 | 正文里有答案（n29 的 1s/14 个就在 4.8.6 里，模型自己引了又否定），但表述与问句不同构 | 判据加固过头：「直接写明」把同义表述、相邻句合并、宽口径涵盖都拒了 | 判据改为「原文里有这个内容」，仍禁止推断与常识补全 |
| q008 | 复跑同一问题即正常作答 | LLM 输出波动，temperature=0 也会一次答一次拒 | 不治；这类抖动靠重复轮次观测，不当成回归 |

修完：命中仍拒答 **15.0% → 2.5%**，端到端 **82.5% → 95.0%**，真幻觉仍是 0%。

引用自校验（`scripts/verify_citations.py`，口径见 [eval/README](eval/README.md)）：

| 结论 | 条数 |
| --- | --- |
| 支持 | 46 |
| 部分 | 13 |
| 不支持 | 0 |

56 → 59 条可核对项是三轮滤噪的结果：第一版把 `GB25201-2010`、`第 1 款`、Markdown 的 `2.`、
引用角标 `[3]` 都当成「关键数值」，74 条里 9 条「不支持」有 8 条是这类假阳性。剩下的 13 条
「部分」都是真的写法差异：跨片段合并（n12 的 `2.0～2.5 倍` 来自两条）、编号粘连（IP55 里的 55）、
单位写法（`150` vs `150mm`）。**0 条「不支持」不等于没有引用问题**，只说明这批答案里没有
「数值凭空出现」的情况。


**「三道口径」是上一轮最大的改动，起因是一个被自己夸大的结论。**
上一轮把 11 道未命中归成「5 道是跨标准同义条文、gold 只许写一个文件」。逐条回库核对后
（`scripts/find_carriers.py` 是为此写的工具）**只有 3 道成立**：

- 成立：n9（GB50116 6.2.6 与 GB50166 逐字同义，连 0.5m 都一样）、n12（2 倍～2.5 倍，连
  「加设缓冲装置」都一样）、n24（通用规范给原则、专用标准给量化下限）
- **不成立**：n20 —— 全库只有 GB51251 一处有那个 10%，GB55036 讲的是余压值 25~50Pa，那是
  **同章节相邻条款的干扰**；q009 —— 规范侧没有任何一片能单独承载完整逻辑，要 4.8.8 与 4.8.12
  两条合起来才等价，那是**多跳**。

把这两道标成「同义承载文档」，就能让宽松口径白涨 5 个点——所以严格/宽松必须并列报，
并且每条 `alternate` 都要在题库的 `note` 里写原文依据。**没有依据的多标注一律视为作弊。**

**第三种失败模式（老表里没有位置的）**：n11 的正确答案稳定排第 5，`k=4` 直接截断，
模型看不到原文就拒答。这类既不进「误拒率」（那是阈值口径），也不进「幻觉率」（那是无答案题），
现在由 `answerable_refused_rate` 单独计量，并拆成「检索没排到 → 拒答」和「排到了仍拒答」两支。
还有第四支：**检索没排到、模型照样给答案**（现在 1 道）——它比拒答更危险，因为端到端作答率
会把它算成成功。**所以端到端 95% 是「答了」的比例，不是「答对」的比例**，两个数别混着用。

`k=4 → 6` 的代价与收益：hit@4 时代 11 道未命中，k=6 后剩 7 道（5 道未命中 + 2 道仅宽松命中），
其中 n11（第 5）、n16（第 7）、n20（第 8）都是纯截断问题——**但 n16 和 n20 在 k=6 仍然没进**，
所以 k 不是无脑加大就有收益，得靠 rerank 把对的排上来。

**阈值为什么定在 0.30 而不是 sweep 推荐的 0.65**：`--sweep` 只拿阈值算 F1，它看不见模型这条路径。
把三档阈值分别跑完整双路评测，结论是反的——

| `MIN_SCORE` | 误拒率 | 阈值拒答 | 模型拒答 | 真幻觉 |
| --- | --- | --- | --- | --- |
| 0.30 | 0% | 0% | 100% | 0% |
| 0.60 | 2.5%（1 道） | 20% | 80% | 0% |
| 0.65 | 10%（4 道） | 50% | 50% | 0% |

阈值从 0.30 抬到 0.65，**幻觉率一点没降**（三种配置都是 0/10），却白赔 4 道本来能答对的题。
原因是这批语料太同质：无答案题的 top1 分数最低 0.577、最高 0.728，和有答案题的分布完全重叠，
余弦分数扛不住「沾边但没这件事」的区分。既然兜住幻觉的是 Prompt，阈值就该只当噪声闸门用，别指望它做判断。

> 这张表是在 `k=4` 时代测的。`k` 改 6 之后重跑，结论方向没变（抬阈值幻觉一例没少），
> 但具体数字会动，下轮补测时再更新。

**改写题戳穿了「纯 BM25 最强」**。50 题主题库偏「术语即答案」——问法与规范用词高度重合，纯 BM25 单档就能 97.5% / MRR 0.8438。新增 12 道改写题（`eval/questions_rewritten.yaml`，用日常说法问同一事实，gold 克隆自主题库不另行标注）后，同一套六档重测：

| 档位 | hit@6 | MRR |
| --- | ---: | ---: |
| 纯向量 | 91.7% | **0.7743** |
| +BM25（加权和） | 91.7% | 0.6898 |
| +RRF（**当前默认**） | 91.7% | 0.6302 |
| 纯 BM25 | **66.7%** | 0.4184 |

纯 BM25 命中率掉 25pp、MRR 腰斩：它当初的强势来自词面重合，不是真检索能力。端到端 91.67%（修正两道语义漂移的题之后，拒答 0%）。**但样本只有 12 道，不足以据此改默认档**——RRF 在 hit@6 上并不吃亏（与纯向量持平），只在 MRR 上被 BM25 的噪声拖低。列为待观察项，扩到 30 道以上再决定。

引用抽检：20 条人工判定 **13 完全支持 / 7 部分支持 / 0 不支持**。7 条「部分支持」里 6 条是拒答题
（只能弱证否，天然拿不到「完全支持」），1 条是答案由条文说明归纳、缺原文直证。
判定时凡涉及数值的都回库核对了被引片段全文，没只看页面截断的 200 字。详见 `eval/results/citation_review.md`。

**无答案题会过期**：扩库前判「文档里没有这个事实」的题，扩库后可能就被推翻了。q012 原本问
「防火卷帘的下降速度标准是多少米每秒」，标注为无答案；加入 GB50877 之后第 22 页明写
「电动启、闭的运行速度应为 2m/min～7.5m/min」，模型据实回答（还纠正了 m/s 与 m/min 的单位），
在统计里却被记成一次幻觉。**每轮扩库后都要复核那 10 道无答案题，不能复用旧标注**——
否则「幻觉率」测的是标注的陈旧程度，不是模型。

> ⚠️ **`hit@4` 被自己的评测脚本坑过一次，值得单独记**：第一版 `resolve_gold` 只把「第一个匹配到的
> 切片」当 gold。但 `chunk_overlap=150` 会让同一句话同时落在相邻两片里，命中另一片就被判成未命中——
> 同一套库、同一批题，把 gold 改成「同文件内含锚点的所有切片」之后是 **55% → 72.5%**，MRR 0.425 → 0.569。
> 教训：指标突然变难看，先怀疑评测代码，再怀疑模型。
>
> 当晚还把这个缺口错记成「服务里是旧索引快照，不重启就对不上新库」。后来专门验了一次：
> 外部进程写入后，**不重载**直接检索，探针片段就排在第 1 名、score 1.000，计数也实时变化
> （chromadb 1.5.9）。也就是说「快照」这个归因不成立——两个毛病同一晚出现，很容易把相关性
> 当成因果。真正的教训是第二层：**改完一个 bug，别顺手给另一个现象编一个听起来合理的原因。**
>
> 换库（改 `CHROMA_DIR`、换集合名、整库替换）时仍然需要重载，现在有 `POST /api/v1/admin/reload`
> 可以免重启，它会返回重载前后的计数，`changed=false` 就说明库本来没变。

**为什么单独开 `/search`**：`/query` 的 citations 已过 `MIN_SCORE`，被阈值拦掉的命中在它眼里等于
「没检索到」，用它算 recall 会把「分低但相关」和「压根没命中」混成一件事。原始排名只有
`/search` 给得出。

**为什么拒答拆三条**：实测 `MIN_SCORE=0.30` 时阈值路径拒答率 **0%**、模型路径 **100%**、
真幻觉率 **0%**——只看 `abstained` 会得出「拒答能力为 0」的错误结论，然后去调一个本来没问题的
阈值。三条一起看才知道：**兜住幻觉的是 Prompt，不是阈值**。抬阈值只换来「少调几次模型」，
代价是误拒一批本来能答的题，不划算。

**为什么 gold 标锚点不标 chunk_id**：`chunk_id` 依赖 `chunk_size`，一调切片整套标注就废。
锚点记成 `(文件, 页码, 原文)`，启动时按三级降级反查当前库（同页 → 同文件 → 全库），
切片口径变了照样能跑；降级匹配单独告警，而不是静默兜底。

**一个锚点要收多个切片，不是一个**：`chunk_overlap` 会让同一句话同时落在相邻两片、跨页的长条款也会重复。
只要切片里含这句原文，它就是「答对所需的片段」，命中任意一片都算检索成功——只认第一片会把重叠算成失败
（实测凭空吃掉 17.5 个点的 `hit@4`）。

产物在 `eval/results/`：带配置指纹的 json、`failures.md`（未命中题 + 实际 top5 + 空归因栏）、
`citation_review.md`（20 条引用抽检样本）。文档数 < 5 或切片 < 100 时脚本直接告警——
那时前 4 名已占全库可观的百分比，指标没有区分度。

## 测试

```bash
pip install -r requirements-dev.txt
pytest            # 99 passed，全部不联网、不消耗 token
```

覆盖：切片元数据与空片段/空页处理、条文化判定与密度阈值、条文/说明分离、去重、中文分词与型号保护、六档融合与深度对齐、`score` 语义（RRF 分不被阈值误砍）、闸门拒答不调模型、重排降级、引用映射、文件名路径穿越、GB18030 解码、上传返回、同内容/同名重复上传、删除（含原文件）、非法类型、伪 PDF、超大文件、`X-API-Key` 鉴权与 CORS、非法档位、缺 Key 降级、`/admin/reload`、跳过重复向量化、Embedding 失败不丢旧数据、首页可访问。

`tests/test_retrieval_merge.py` 专门钉住合并两版代码时的 4 处裁决，以及两个**不会报错的**静默错误：`SearchRequest` 缺 `mode` 字段被 Pydantic 静默忽略（六档跑出同一组数）、`eval.py` 的 `first_rank` 参数顺序相反。这类错误不会抛异常，只会让消融矩阵和评测悄悄失真。

## 从旧版迁移

旧版接口为 `/upload`、`/query`，向量集合是默认的 `langchain`。新版统一到 `/api/v1/*`，集合改为 `kb_v2`：

- 旧库数据不自动迁移，直接重建：`python scripts/rebuild_index.py --apply --collection kb_v2`
- `.env` 保持原有两个 Key 即可，无需改动；建议顺手填上 `API_KEY`
- 切片口径从定长改成按条，但 **`gold` 锚点不用重标**：锚点记的是 `(文件, 页码, 原文)`，启动时三级降级反查，换口径照样能跑

## 已知限制与可扩展方向

- 单进程 + 本地 Chroma，适合单机/演示；要多副本需换成 Chroma Server 或 pgvector
- 只支持文本型 PDF，扫描件需先 OCR（本身 degraded 到只剩少量文本层的 PDF 往往能成功入库，但召回会很差——这类文档最好提前检测并直接报错）
- `API_KEY` 是单 key 的轻量鉴权，适合自用/内网；做多人使用应换成 OAuth/JWT 并加审计
- 未做用户体系与多知识库隔离（可按 `doc_id` 前缀或 Chroma `where` 加 `tenant` 字段扩展）
- **重排档不能用于交互**：20 条候选 × 640 窗口，p95 约 11.5s。真解法是「先抽与问题最相关的句子再重排」，成本能降一个量级——下一阶段。线上 400ms 预算下重排覆盖率为 0，闸门会整档跳过，所以默认档不带它
- **`RRF_K=60` 没扫过**：调它大概率能再挤 1~2 个点，但在 40 道题上调就是过拟合，暂不做
- **改写题只有 12 道**：足以戳穿「纯 BM25 最强」，但不足以据此换默认档。扩到 30 道以上再定
- **BM25 没做词权重**：条款号（`5.1.1`）和普通词同权。规范语料里条款号是强信号，值得单独提权
- **无答案题只有 10 道**：0% 幻觉率的 95% 置信区间极宽，只能说「这批题没漏」，不能说「不会幻觉」。而且扩库后旧标注会过期，每轮都要复核
- **检索侧已不是瓶颈**：四档非重排档 hit@20 全 100%，40 道题只剩 1 道未命中；再往上的空间在答案侧（表述差异导致的拒答）
- **单进程 + 本地 Chroma**：适合单机/演示；要多副本需换成 Chroma Server 或 pgvector
- **只支持文本型 PDF**：扫描件需先 OCR。本身 degraded 到只剩少量文本层的 PDF 往往能成功入库但召回很差，这类文档最好提前检测并直接报错
- **`API_KEY` 是单 key 的轻量鉴权**：适合自用/内网；做多人使用应换成 OAuth/JWT 并加审计
- **未做用户体系与多知识库隔离**：可按 `doc_id` 前缀或 Chroma `where` 加 `tenant` 字段扩展
- **WSL 侧起服务受限**：Ubuntu 自带 Python 3.14，torch 还没有对应轮子，所以本机是在 Windows 侧跑的（向量库放 `D:` 本地盘，WSL 侧通过 `/mnt/d` 也能读）
- 可加：SSE 流式输出、查询改写（HyDE / 多查询）、条文号提权、会话记忆
