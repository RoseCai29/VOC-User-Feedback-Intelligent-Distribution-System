# VOC User Feedback Intelligent Distribution System

[![CI](https://github.com/RoseCai29/VOC-User-Feedback-Intelligent-Distribution-System/actions/workflows/ci.yml/badge.svg)](https://github.com/RoseCai29/VOC-User-Feedback-Intelligent-Distribution-System/actions/workflows/ci.yml)

**English** | [中文](#voc-用户反馈智能分发系统)

An LLM agent system that classifies, deduplicates, and routes user feedback (VOC) using a **LangGraph ReAct loop** with **OpenAI Function Calling**. Built with FastAPI, ChromaDB evidence retrieval, and a zero-key deterministic fallback mode.

## Highlights

- **ReAct decision loop** — `agent_decision → tool_executor → agent_reflection`, the LLM picks the next tool each turn and decides when to converge (not a fixed pipeline).
- **6 Function Calling tools** — `classify` (3-level classification + confidence), `retrieve_evidence` (ChromaDB semantic search), `dedup` (n-gram Jaccard), `route` (product-line × category lookup table), `push_service_guide`, `crawl_feedback`.
- **Low-confidence self-correction** — confidence < 0.7 triggers evidence retrieval and re-classification, up to 3 rounds.
- **Emoji-aware preprocessing** — emojis are stripped before classification while a weak-signal profile (strong-negative / positive / ambiguous counts) is kept in state; emoji-only text is passed through untouched for the LLM to judge. No forced sentiment polarity.
- **Dual mode** — full LLM agent mode, or `USE_DETERMINISTIC=true` rule-engine mode that runs with **no API key** (same graph, deterministic tool cascade).
- **Graceful degradation** — LLM failure falls back to the rule engine; ChromaDB unavailable falls back to Jaccard similarity.
- **Production guards** — bounded trace length, input length limits, bounded dedup buckets, thread-safe dedup cache, structured logging.
- **FastAPI service** — `POST /feedback`, `GET /feedback/{id}`, `/health`, with SQLite persistence via SQLAlchemy 2.0.

## Architecture

```
POST /feedback
      │
      ▼
┌──────────────────────────────────────────────────────┐
│  fast path (LLM mode only)                           │
│  small model dual-threshold gate 0.80 / 0.75         │
│  passes → whole pipeline runs deterministically,     │
│           zero LLM calls                             │
└──────────────┬───────────────────────────────────────┘
               │ gate not met / rules mode
               ▼
┌──────────────────────────────────────────────────────┐
│  preprocess   emoji strip + weak-signal profile      │
│               (strong-negative / positive /          │
│                ambiguous counts kept in state;       │
│                emoji-only text kept verbatim)        │
└──────────────┬───────────────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────────┐
│  agent_decision   LLM reads state → picks 1 tool    │
└──────────────┬───────────────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────────┐
│  tool_executor    classify / retrieve_evidence /     │
│                   dedup / route /                    │
│                   push_service_guide / crawl_feedback│
└──────────────┬───────────────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────────┐
│  agent_reflection confidence < 0.7 ?                 │
│                   → re-classify with evidence (≤3轮) │
│                   → else continue / FINISHED         │
└──────────────┬───────────────────────────────────────┘
               │
        loop ──┴──► END ──► SQLite persist + response
```

**Classification**: 作品建议 (suggestion) / 舆情吐槽 (sentiment) / 售后报修 (after-sales), each with sub-categories, tags, and a confidence score.

**Routing**: deterministic 2-D lookup table `(source × category × sub-category) → product line / assignee / SLA hours` — zero LLM involvement, zero hallucination on the routing step.

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env          # fill OPENAI_API_KEY (or see below)

# 1) CLI demo (no server needed)
python -m src.main

# 2) Run with zero API key: deterministic rule mode
#    set USE_DETERMINISTIC=true in .env

# 3) API server
uvicorn src.api:app --port 8000
# docs at http://localhost:8000/docs
```

Example request:

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"raw_text": "APP经常闪退崩溃，打开就黑屏", "source": "社区"}'
```

Response includes `category / sub_category / confidence / tags / route_assignee / guide_phone / retry_count / trace_steps / elapsed_ms / mode`.

## Tests & Evaluation

```bash
python -m pytest tests/ -q          # 55 tests, no API key required

# synthetic labeled eval set (22 examples)
python -m src.evaluate --dataset data/eval_set.json --mode rules
python -m src.evaluate --dataset data/eval_set.json --mode semantic

# real App Store reviews (2171 raw, 269 human-labeled, dev 116 / test 153)
python -m src.evaluate --dataset data/real/eval_real_dev.jsonl
python -m src.evaluate --dataset data/real/eval_real_test.jsonl
python -m scripts.eval_agent --dataset data/real/eval_real_test.jsonl --out data/real/agent_metrics_rules.json
```

The evaluator reports level-1 / level-2 accuracy, bootstrap 95% CI, macro F1, per-category precision-recall-F1, and a confusion matrix.

Measured on 153 real test reviews (rules mode): level-1 **74.5%**, level-2 **66.7%**, macro F1 **0.73**
(synthetic set: 86.4% / 77.3% / 0.86 — a 12pt gap that quantifies synthetic-data optimism).
Agent-level metrics: pipeline success 100%, degraded 0%, escalation 15.0%, avg 12.1 steps, avg latency 164ms.

After one annotation-flywheel retrain round (see below), the small model's threshold-gated fast path measures **85.2% L1 / 75.3% L2 @ 52.9% coverage** on the same held-out test set (pre-retrain: 53.5% / 42.3% — a domain-shift negative optimization). LLM-mode end-to-end with a local 3B model: fast path fires on 52.9% of cases with zero LLM calls, pipeline success 100%, 72.6% L1 overall, avg 2.6s/case. Cascade (retrained model + rules fallback) reaches 72.6% L1 — still below pure rules (74.5%), so the defaults stay `CLASSIFIER_MODE=rules` with the fast path enabled for LLM mode.

CI runs on every push/PR via GitHub Actions (`.github/workflows/ci.yml`): fatal-error lint (`ruff --select E9,F63,F7,F82`) + the full pytest suite.

Annotation flywheel (one full loop): low-confidence escalations land in the pending pool → humans approve labels (`update_pending_label`) → approved samples merge back into training data (`data/voc_train_v3.jsonl`, with an explicit test-set leak check) → retrain (`python -m scripts.train_from_annotation --input data/voc_train_v3.jsonl`) → recalibrate thresholds on dev (`python -m scripts.calibrate_thresholds`) → re-gate on the held-out test set.

## Configuration

| Env var | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | `sk-placeholder` | API key (not needed in deterministic mode) |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Any OpenAI-compatible endpoint |
| `MODEL_NAME` | `gpt-4o-mini` | Chat model |
| `USE_DETERMINISTIC` | `false` | `true` = rule engine, zero LLM calls |
| `CONFIDENCE_THRESHOLD` | `0.7` | Low-confidence self-correction trigger |
| `MAX_RETRIES` | `3` | Max evidence-retrieval re-classify rounds |
| `DEDUP_THRESHOLD` | `0.75` | Jaccard similarity threshold |
| `DATABASE_URL` | `sqlite:///./data/voc.db` | Persistence |
| `CHROMA_PERSIST_DIR` | `./data/chroma` | Vector store path |
| `SKIP_CHROMADB` | — | `1` = skip ChromaDB, use Jaccard directly |
| `ENABLE_LIVE_CRAWL` | `false` | Live crawling switch |
| `CLASSIFIER_MODE` | `rules` | `rules` \| `semantic` \| `small` \| `cascade` — retrained cascade (rules fallback) still measures 72.6% L1 vs rules 74.5%, so rules stays default |
| `CASCADE_FALLBACK` | `llm` | Fallback engine when the small model is low-confidence in cascade mode |
| `ENABLE_FAST_PATH` | `true` | LLM-mode fast pipeline; retrained model gates at 85.2% L1 / 75.3% L2 @ 52.9% coverage (was 53.5% pre-retrain) |
| `SMALL_MODEL_PATH` | `./data/models/bert_voc_v2` | Retrained BERT (real-label flywheel); `bert_voc` is the pseudo-label predecessor |
| `SMALL_MODEL_CATEGORY_THRESHOLD` | `0.80` | Level-1 direct-output threshold, verified 85.2% precision on held-out test |
| `SMALL_MODEL_SUB_THRESHOLD` | `0.75` | Level-2 direct-output threshold, verified 75.3% precision on held-out test |
| `ENABLE_PENDING_POOL` | `true` | Write low-confidence samples to the human-annotation pool |

## Project Structure

```
src/
├── graph.py      # LangGraph state machine: decision → execute → reflect
├── tools.py      # 6 function-calling tools + OpenAI tool schemas + route table
├── state.py      # FeedbackState (LangGraph state) + pydantic models + enums
├── llm.py        # OpenAI-compatible client, JSON completion with fallback
├── api.py        # FastAPI endpoints (event-loop-safe via asyncio.to_thread)
├── db.py         # SQLAlchemy 2.0 models: FeedbackRecord / DedupBucket / PendingLabel
├── evaluate.py   # accuracy / bootstrap CI / macro-F1 / confusion matrix
├── classifier/   # small-model cascade: dataset, linear/bert, thresholds, bootstrap
├── preprocess.py # emoji / noise normalization
├── config.py     # pydantic-settings config center + production guards
└── main.py       # CLI demo with 8 sample feedbacks
scripts/
├── fetch_real_reviews.py   # App Store CN RSS → data/real/reviews_raw.jsonl
├── build_real_eval.py      # label queue + manual labels → stable dev/test split
├── ingest_real_reviews.py  # idempotent batch ingestion into SQLite
├── train_from_annotation.py # retrain BERT from annotation files (flywheel)
├── calibrate_thresholds.py # dual-threshold grid scan on a holdout split
└── eval_agent.py           # end-to-end agent metrics (steps / degradation / tokens)
data/eval_set.json    # 22 synthetic labeled examples
data/real/            # real review corpus, human labels, dev/test eval sets, metrics
tests/                # 55 tests: graph loop, tools, classifier, eval-set sanity
```

## Known Limitations

- **No authentication** on the API — acceptable for a demo; production needs API key / JWT.
- **Real eval set is small** (269 human-labeled reviews out of 2171 collected), and the 4-category label schema is a judgment call — level-2 accuracy on real data is the weakest link.
- **Crawler is a skeleton** — `crawl_feedback` returns mock items unless live crawling is enabled; App Store RSS collection lives in `scripts/fetch_real_reviews.py` rather than the runtime tool.
- **Deterministic mode is keyword-based** — it proves the degradation path, not semantic understanding; after rule expansion it reaches 74.5% L1 / 66.7% L2 on real data, while a local 3B LLM reaches 71.2% L1 / 45.8% L2. The LLM leg's value depends on model strength: a 3B local model is not a proxy for a production model, so the measured numbers argue for the cascade + degrade design rather than for LLM superiority.
- **Domain shift bit us once** — the first small model was trained entirely on pseudo-labeled synthetic text (0 manual labels); on real reviews its threshold-gated outputs scored only 53.5% L1 / 42.3% L2, making cascade a 13pt *negative* optimization (61.4% vs rules 74.5%). We retrained with the annotation flywheel (116 real dev labels + 14 approved pending samples merged into training, test set leak-checked): the retrained model gates at **85.2% L1 / 75.3% L2 @ 52.9% coverage** on the held-out test set. Cascade with a rules fallback still measures 72.6% L1 (hard samples are hard for both engines), so `CLASSIFIER_MODE` stays `rules` by default — honest numbers over narrative.
- Single-process SQLite + in-memory dedup cache — no horizontal scaling.

## Tech Stack

LangGraph · FastAPI · OpenAI Function Calling · ChromaDB · SQLAlchemy 2.0 · Pydantic v2 · Docker

---

# VOC 用户反馈智能分发系统

[English](#voc-user-feedback-intelligent-distribution-system) | **中文**

基于 **LangGraph ReAct 循环** + **OpenAI Function Calling** 的用户反馈（VOC）智能分类、去重与路由分发系统。技术栈：FastAPI、ChromaDB 证据检索、零 Key 确定性降级模式。

## 核心特性

- **ReAct 决策循环** —— `agent_decision → tool_executor → agent_reflection`，LLM 每轮自主选择下一个工具并决定何时收敛（非固定流水线）。
- **6 个 Function Calling 工具** —— `classify`（三级分类+置信度）、`retrieve_evidence`（ChromaDB 语义检索）、`dedup`（n-gram Jaccard 去重）、`route`（产品线×分类倒排路由表）、`push_service_guide`（售后电话/链接推送）、`crawl_feedback`（异步爬虫骨架）。
- **低置信自纠错** —— 置信度 < 0.7 自动检索历史证据并重分类，最多 3 轮。
- **Emoji 感知预处理** —— 分类前剥离表情，仅保留弱信号三分组统计（强负/正/歧义计数）入状态；纯表情文本保留原文交给 LLM 判断，不强行判极性。
- **双模式** —— 完整 LLM Agent 模式 / `USE_DETERMINISTIC=true` 确定性规则模式（**无需任何 API Key**，同一张图，规则引擎级联决策）。
- **优雅降级** —— LLM 调用失败自动降级规则引擎；ChromaDB 不可用自动降级 Jaccard 相似度检索。
- **工程护栏** —— trace 轨迹有界、输入长度上限、去重桶成员上限、去重缓存线程安全、结构化日志（含全量 traceback）。
- **FastAPI 服务** —— `POST /feedback`、`GET /feedback/{id}`、`/health`，SQLAlchemy 2.0 + SQLite 持久化，`asyncio.to_thread` 避免阻塞事件循环。

## 架构

```
POST /feedback
      │
      ▼
┌──────────────────────────────────────────────────────┐
│  fast path（仅 LLM 模式）                             │
│  小模型双阈值门禁 0.80 / 0.75                         │
│  达标 → 整链路确定性直通，零 LLM 调用                 │
└──────────────┬───────────────────────────────────────┘
               │ 未达标 / 规则模式
               ▼
┌──────────────────────────────────────────────────────┐
│  预处理     emoji 剥离 + 弱信号三分组统计             │
│             （强负/正/歧义计数入状态；                │
│               纯表情文本保留原文）                    │
└──────────────┬───────────────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────────┐
│  agent_decision   LLM 读取状态 → 选择 1 个工具      │
└──────────────┬───────────────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────────┐
│  tool_executor    classify / retrieve_evidence /     │
│                   dedup / route /                    │
│                   push_service_guide / crawl_feedback│
└──────────────┬───────────────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────────┐
│  agent_reflection 置信度 < 0.7 ?                     │
│                   → 取证后重分类（≤3 轮）            │
│                   → 否则 continue / FINISHED         │
└──────────────┬───────────────────────────────────────┘
               │
        loop ──┴──► END ──► SQLite 落库 + 返回响应
```

**三级分类**：作品建议 / 舆情吐槽 / 售后报修，各含二级分类、标签与置信度。

**路由分发**：确定性二维倒排表 `(来源 × 一级分类 × 二级分类) → 产品线 / 负责人 / SLA 时长`——路由环节零 LLM 参与、零幻觉。

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env          # 填入 OPENAI_API_KEY（或用下方零 Key 模式）

# 1) CLI 演示（无需起服务）
python -m src.main

# 2) 零 API Key 运行：确定性规则模式
#    在 .env 中设 USE_DETERMINISTIC=true

# 3) API 服务
uvicorn src.api:app --port 8000
# 文档: http://localhost:8000/docs
```

请求示例：

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"raw_text": "APP经常闪退崩溃，打开就黑屏", "source": "社区"}'
```

响应包含 `category / sub_category / confidence / tags / route_assignee / guide_phone / retry_count / trace_steps / elapsed_ms / mode`。

## 测试与评测

```bash
python -m pytest tests/ -q          # 55 个测试，无需 API Key

# 合成标注集（22 条，脚本生成）
python -m src.evaluate --dataset data/eval_set.json --mode rules
python -m src.evaluate --dataset data/eval_set.json --mode semantic

# 真实数据：App Store 中国区评论 2171 条原始 / 269 条人工标注（dev 116 / test 153）
python -m src.evaluate --dataset data/real/eval_real_dev.jsonl
python -m src.evaluate --dataset data/real/eval_real_test.jsonl
python -m scripts.eval_agent --dataset data/real/eval_real_test.jsonl --out data/real/agent_metrics_rules.json

# 真实评论端到端入库（幂等：同一 source+review_id 不新增记录）
python -m scripts.ingest_real_reviews --limit 200
```

评测输出 level-1 / level-2 准确率、bootstrap 95% 置信区间、宏观 F1、各类别精确率-召回率-F1、混淆矩阵。

真实 test 153 条（规则模式）：一级 **74.5%**、二级 **66.7%**、宏观 F1 **0.73**；
合成集 86.4% / 77.3% / 0.86 —— 约 12 个点的落差量化了合成数据的乐观偏差。
Agent 层指标：链路成功率 100%、降级率 0%、升级率 15.0%、平均 12.1 步、平均延迟 164ms。

完成一轮标注飞轮回流重训（见下文）后，小模型双阈值 fast path 在同一留出 test 集上实测 **85.2% L1 / 75.3% L2 @ 覆盖 52.9%**（重训前 53.5% / 42.3%——域偏移导致的负优化）。LLM 模式本地 3B 端到端：52.9% 的样本零 LLM 调用直通，链路成功率 100%，整体 L1 72.6%，平均 2.6s/条。级联（重训模型+规则兜底）72.6% L1，仍低于纯规则 74.5%，因此默认保持 `CLASSIFIER_MODE=rules`、LLM 模式开启 fast path。

CI 由 GitHub Actions 在每次 push/PR 触发（`.github/workflows/ci.yml`）：致命错误 lint（`ruff --select E9,F63,F7,F82`）+ 全量 pytest。

标注飞轮（完整跑通一轮）：低置信升级样本进待标注池 → 人工审批打标（`update_pending_label`）→ 审批样本回流训练集（`data/voc_train_v3.jsonl`，内置 test 集泄漏断言）→ 重训（`python -m scripts.train_from_annotation --input data/voc_train_v3.jsonl`）→ dev 上重标定阈值（`python -m scripts.calibrate_thresholds`）→ 留出 test 集守门复验。

## 配置项

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `OPENAI_API_KEY` | `sk-placeholder` | API Key（确定性模式不需要） |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | 任意 OpenAI 兼容端点 |
| `MODEL_NAME` | `gpt-4o-mini` | 对话模型 |
| `USE_DETERMINISTIC` | `false` | `true` = 规则引擎，零 LLM 调用 |
| `CONFIDENCE_THRESHOLD` | `0.7` | 低置信自纠错触发阈值 |
| `MAX_RETRIES` | `3` | 取证重分类最大轮数 |
| `DEDUP_THRESHOLD` | `0.75` | Jaccard 相似度阈值 |
| `DATABASE_URL` | `sqlite:///./data/voc.db` | 持久化 |
| `CHROMA_PERSIST_DIR` | `./data/chroma` | 向量库路径 |
| `SKIP_CHROMADB` | — | `1` = 跳过 ChromaDB，直接 Jaccard |
| `ENABLE_LIVE_CRAWL` | `false` | 真实爬取开关 |
| `CLASSIFIER_MODE` | `rules` | `rules` \| `semantic` \| `small` \| `cascade` —— 重训后级联(规则兜底)实测 72.6% L1 仍低于规则 74.5%，默认保持 rules |
| `CASCADE_FALLBACK` | `llm` | 级联模式下小模型低置信时的兜底引擎 |
| `ENABLE_FAST_PATH` | `true` | LLM 模式快速直通；重训后双阈值直出 85.2% L1 / 75.3% L2 @ 覆盖 52.9%（重训前仅 53.5%） |
| `SMALL_MODEL_PATH` | `./data/models/bert_voc_v2` | 真实标注重训的 BERT；`bert_voc` 为伪标签前身模型 |
| `SMALL_MODEL_CATEGORY_THRESHOLD` | `0.80` | 一级直出阈值，留出 test 集验证精度 85.2% |
| `SMALL_MODEL_SUB_THRESHOLD` | `0.75` | 二级直出阈值，留出 test 集验证精度 75.3% |
| `ENABLE_PENDING_POOL` | `true` | 低置信样本写入待标注池 |

## 项目结构

```
src/
├── graph.py      # LangGraph 状态机: 决策 → 执行 → 反思
├── tools.py      # 6 个 function-calling 工具 + tool schema + 倒排路由表
├── state.py      # FeedbackState (LangGraph 状态) + pydantic 模型 + 枚举
├── llm.py        # OpenAI 兼容客户端，JSON 补全 + 解析降级
├── api.py        # FastAPI 端点（asyncio.to_thread 防阻塞）
├── db.py         # SQLAlchemy 2.0 模型: FeedbackRecord / DedupBucket / PendingLabel
├── evaluate.py   # 准确率 / bootstrap 置信区间 / 宏观 F1 / 混淆矩阵
├── classifier/   # 小模型级联: 数据集、linear/bert、阈值标定、bootstrap
├── preprocess.py # emoji / 噪声归一化
├── config.py     # pydantic-settings 配置中心 + 生产护栏
└── main.py       # CLI 演示，8 条样例反馈
scripts/
├── fetch_real_reviews.py   # App Store 中国区 RSS → data/real/reviews_raw.jsonl
├── build_real_eval.py      # 标注队列 + 人工标注 → 稳定 dev/test 切分
├── ingest_real_reviews.py  # 批量幂等入库 SQLite
├── train_from_annotation.py # 基于标注数据重训 BERT（标注飞轮）
├── calibrate_thresholds.py # 留出集双阈值网格标定
└── eval_agent.py           # Agent 端到端指标（步数/降级/token 成本）
data/eval_set.json    # 22 条合成标注样例
data/real/            # 真实评论语料、人工标注、dev/test 评测集与指标
tests/                # 55 个测试: 图循环、工具、分类器、评测集校验
```

## 已知边界与短板

- **API 无鉴权** —— 演示项目可接受；生产需 API Key / JWT。
- **真实评测集偏小**（2171 条采集里人工标注 269 条），且四级分类标注本身有判断空间 —— 真实数据上二级准确率是最弱环节。
- **爬虫是骨架** —— `crawl_feedback` 默认返回 mock 数据；真实 App Store 评论采集放在 `scripts/fetch_real_reviews.py`，不在运行时工具里（合规考虑）。
- **确定性模式基于关键词** —— 它证明的是降级链路可用，不是语义理解能力；规则扩充后真实数据上一级 74.5% / 二级 66.7%，本地 3B 模型一级 71.2% / 二级 45.8%。LLM 支路的价值取决于模型强度：本地 3B 不能代表生产级模型，这组数字支撑的是「级联 + 降级」设计，而非「LLM 一定更强」。
- **域偏移踩过一次坑** —— 初代小模型全用伪标签合成文本训练（人工标签 0 条），真实评论上双阈值直出仅 53.5% L1 / 42.3% L2，导致级联成为 13 个点的**负优化**（61.4% vs 规则 74.5%）。通过标注飞轮回流（116 条 dev 真实标注 + 14 条审批池样本并入训练，内置 test 集泄漏断言）重训后，留出 test 集上双阈值直出达 **85.2% L1 / 75.3% L2 @ 覆盖 52.9%**；但级联（规则兜底）端到端实测 72.6% L1 仍略低于纯规则——困难样本对两个引擎都困难，因此 `CLASSIFIER_MODE` 默认保持 `rules`，如实报数不追叙事。
- 单进程 SQLite + 内存去重缓存 —— 不支持水平扩展。

## 技术栈

LangGraph · FastAPI · OpenAI Function Calling · ChromaDB · SQLAlchemy 2.0 · Pydantic v2 · Docker

