# VOC User Feedback Intelligent Distribution System

**English** | [中文](#voc-用户反馈智能分发系统)

An LLM agent system that classifies, deduplicates, and routes user feedback (VOC) using a **LangGraph ReAct loop** with **OpenAI Function Calling**. Built with FastAPI, ChromaDB evidence retrieval, and a zero-key deterministic fallback mode.

## Highlights

- **ReAct decision loop** — `agent_decision → tool_executor → agent_reflection`, the LLM picks the next tool each turn and decides when to converge (not a fixed pipeline).
- **6 Function Calling tools** — `classify` (3-level classification + confidence), `retrieve_evidence` (ChromaDB semantic search), `dedup` (n-gram Jaccard), `route` (product-line × category lookup table), `push_service_guide`, `crawl_feedback`.
- **Low-confidence self-correction** — confidence < 0.7 triggers evidence retrieval and re-classification, up to 3 rounds.
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
python -m pytest tests/ -q          # 25 tests, no API key required

# classification accuracy on the labeled eval set (22 examples)
python -m src.evaluate --mode rules      # keyword baseline
python -m src.evaluate --mode semantic   # LLM classifier
```

The evaluator reports level-1 / level-2 accuracy, macro F1, per-category precision-recall-F1, and a confusion matrix.

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

## Project Structure

```
src/
├── graph.py      # LangGraph state machine: decision → execute → reflect
├── tools.py      # 6 function-calling tools + OpenAI tool schemas + route table
├── state.py      # FeedbackState (LangGraph state) + pydantic models + enums
├── llm.py        # OpenAI-compatible client, JSON completion with fallback
├── api.py        # FastAPI endpoints (event-loop-safe via asyncio.to_thread)
├── db.py         # SQLAlchemy 2.0 models: FeedbackRecord / DedupBucket / AgentTrace
├── evaluate.py   # accuracy / macro-F1 / confusion matrix on labeled set
├── config.py     # pydantic-settings config center + production guards
└── main.py       # CLI demo with 8 sample feedbacks
data/eval_set.json    # 22 labeled examples (independent of the implementation)
tests/                # 25 tests: graph loop, tools, eval-set sanity
```

## Known Limitations

- **No authentication** on the API — acceptable for a demo; production needs API key / JWT.
- **Eval set is synthetic** (22 examples generated by a script), not real production feedback.
- **Crawler is a skeleton** — `crawl_feedback` returns mock items unless live crawling is enabled; real Xiaomi community crawling was intentionally not shipped.
- **Deterministic mode is keyword-based** — it proves the degradation path, not semantic understanding; use LLM mode for real workloads.
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
python -m pytest tests/ -q          # 25 个测试，无需 API Key

# 标注集上的分类准确率（22 条样例）
python -m src.evaluate --mode rules      # 关键词基线
python -m src.evaluate --mode semantic   # LLM 语义分类
```

评测输出 level-1 / level-2 准确率、宏观 F1、各类别精确率-召回率-F1、混淆矩阵。

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

## 项目结构

```
src/
├── graph.py      # LangGraph 状态机: 决策 → 执行 → 反思
├── tools.py      # 6 个 function-calling 工具 + tool schema + 倒排路由表
├── state.py      # FeedbackState (LangGraph 状态) + pydantic 模型 + 枚举
├── llm.py        # OpenAI 兼容客户端，JSON 补全 + 解析降级
├── api.py        # FastAPI 端点（asyncio.to_thread 防阻塞）
├── db.py         # SQLAlchemy 2.0 模型: FeedbackRecord / DedupBucket / AgentTrace
├── evaluate.py   # 标注集准确率 / 宏观 F1 / 混淆矩阵
├── config.py     # pydantic-settings 配置中心 + 生产护栏
└── main.py       # CLI 演示，8 条样例反馈
data/eval_set.json    # 22 条标注样例（独立于实现标注）
tests/                # 25 个测试: 图循环、工具、评测集校验
```

## 已知边界与短板

- **API 无鉴权** —— 演示项目可接受；生产需 API Key / JWT。
- **评测集是合成的**（脚本生成的 22 条），非真实生产反馈数据。
- **爬虫是骨架** —— `crawl_feedback` 默认返回 mock 数据，未内置真实社区爬取（合规考虑）。
- **确定性模式基于关键词** —— 它证明的是降级链路可用，不是语义理解能力；真实场景请用 LLM 模式。
- 单进程 SQLite + 内存去重缓存 —— 不支持水平扩展。

## 技术栈

LangGraph · FastAPI · OpenAI Function Calling · ChromaDB · SQLAlchemy 2.0 · Pydantic v2 · Docker
