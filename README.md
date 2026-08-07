# icore — 企业级 LLM 工作流编排平台

**Coding-Driven Workflow Orchestration Platform**

icore 是一个企业级工作流能力平台，通过编码形式将大模型（LLM）驱动的任务编排成可复用、可调度的工作流。不提供拖拽式的可视化编辑器——工作流完全通过 Python 代码定义，充分利用编程语言的表达能力（条件分支、循环、异常处理、子流程嵌套）实现复杂业务逻辑。

---

## 核心特性

- **DAG 工作流引擎**：基于有向无环图的任务调度，支持拓扑排序、并行执行、子工作流嵌套（工作流调度工作流）
- **任务抽象层**：`BaseTask` 抽象基类，每个 Task 有强类型的输入/输出 Pydantic 模型，`TaskRegistry` 注册与管理
- **多模型管理**：配置多个 OpenAI 兼容模型（GPT-4o、Claude、DeepSeek、本地模型等），自动路由 + 故障转移 + 成本感知选择；v0.5 起支持 **多模态视觉模型**（GPT-4V / Claude Vision 等）
- **多数据库支持**：统一 `BaseConnector` 抽象，适配器模式支持 PostgreSQL、Oracle、Hive、MySQL 四类数据源，异步连接池
- **向量数据库集成（v0.5）**：`BaseVectorStore` 抽象 + Milvus 适配器，支持 RAG 检索增强生成、语义搜索、过滤表达式
- **图引擎与知识图谱（v0.5）**：`BaseGraphStore` 抽象 + Neo4j 适配器，支持实体关系图谱构建、Cypher 查询、GraphRAG 多跳推理
- **多模态文件处理（v0.5）**：`MediaFile` 统一抽象 + Image/Video/Audio 三类处理器，支持 OCR、关键帧提取、ASR 语音转文字
- **统一 API**：仅两个端点——`GET /health`（多组件健康探针）和 `POST /invoke`（主调用接口，含幂等键去重），Swagger 自动生成
- **流式返回**：支持 `stream=true` 的 SSE（Server-Sent Events）流式输出，适合 LLM 逐 token 生成场景
- **回调机制**：异步结果投递到 `callback_url`，解耦请求方
- **工程健壮性（v0.5）**：统一异常分级体系（`ICoreError` 基类 + code/http_status/retryable）、熔断器（CLOSED/OPEN/HALF_OPEN 状态机）、模型降级链、分布式锁（Redis/Memory）、幂等键缓存、DB 层乐观锁
- **高并发处理**：全 asyncio 架构、内存/Redis 双模式任务队列、Token Bucket 限流、按工作流/任务类型的并发控制、背压机制（已接入 API 层，过载返回 503）
- **优雅关闭（v0.5）**：lifecycle hooks 自动取消后台任务、按顺序关闭 model/db/vectorstore/graphstore 等所有外部连接
- **服务暴露**：工作流可封装为 MCP 服务、Tool 服务（函数调用）、Streamlit 应用、SSE 端点——一套工作流多协议消费
- **依赖注入**：`TaskContext` 注入模型适配器、数据库连接、向量库、图库、媒体处理器、分布式锁、熔断器、元数据，Task 不硬编码依赖，可测试、可替换
- **可观测性体系（v0.6）**：Prometheus Metrics（`/metrics` 端点，10+ 自定义指标）+ OpenTelemetry Tracing（贯穿 invoke → DAG → model call 全链路）+ 结构化日志（JSON 输出，可推送 Loki）
- **多 Agent 协作框架（v0.6）**：Agent 节点作为 DAG 一等公民，支持 REACT / SUPERVISOR / SWARM 三种协作模式，硬限制约束（max_iterations / max_tool_calls），available_tasks 复用已注册普通 Task
- **LLM 语义缓存（v0.6）**：L1 精确匹配（md5）+ L2 向量相似（cosine ≥ 0.95）两级缓存，命中 < 10ms、零 token 成本
- **工作流持久化与断点续跑（v0.6）**：`workflow_executions` + `task_executions` 表持久化到 PostgreSQL，支持 checkpoint 续跑、审计日志、历史查询
- **消息队列触发器（v0.6）**：Kafka / RabbitMQ 消费消息自动执行工作流，事件驱动 DAG
- **鉴权与多租户（v0.6）**：API Key + JWT（RS256）双模式鉴权，RBAC 三角色（admin / developer / viewer），`/health` 免鉴权
- **混合检索与 Reranking（v0.6）**：Dense（向量 ANN）+ Sparse（BM25）+ RRF 融合 + Cross-Encoder 重排序
- **Prompt 模板管理与版本化（v0.6）**：YAML 模板 + 版本切换 + A/B 测试（50/50 分流，指标入 Prometheus）
- **对象存储集成（v0.6）**：`BaseObjectStore` 抽象 + `MinIOAdapter` / `InMemoryObjectStore`，预签名 URL 下载、客户端直传、大文件分片上传
- **结构化弹性（v0.6）**：4 种退避策略（CONSTANT / LINEAR / EXPONENTIAL / EXPONENTIAL_JITTER）+ 死信队列（持久化到 PG、可 replay）+ Saga 补偿事务（自动回滚）
- **全链路背压（v0.6）**：`BackpressureCoordinator` 为 LLM / Milvus / Neo4j / PG / 内存预算分别配置并发与限流阈值，`/health` 暴露各组件饱和状态
- **配置热加载（v0.6）**：`HotReloadCoordinator` 监听 `models.yaml` / `prompts/*.yaml` 变化，watchdog / polling 双模式，无需重启
- **优雅降级矩阵（v0.6）**：`GracefulDegradationCoordinator` 主备 provider 自动切换（LLM→fallback model、Milvus→InMemory、Neo4j→InMemory、Redis→MemoryLock、PG→本地文件），健康探针自动恢复
- **安全加固（v0.6）**：Prompt 注入检测（关键词 + 启发式 + 模型三层）+ PII 脱敏（手机/身份证/邮箱/API Key）+ 可配置开关
- **容器化一键部署（v0.6）**：`docker-compose.full.yaml` 一键拉起 icore + PG + Redis + Milvus + Neo4j + MinIO；`docker-compose.obs.yaml` 拉起 Prometheus + Grafana + Jaeger + Loki；Helm Chart 支持 K8s

---

## 技术栈

| 层面 | 技术 | 用途 |
|------|------|------|
| Web 框架 | **FastAPI** | 原生异步、自动 Swagger、Pydantic 集成 |
| 异步运行时 | **asyncio + uvicorn** | 高并发 I/O 密集场景 |
| 数据校验 | **Pydantic v2** | 类型安全、自动校验、Schema 生成 |
| 关系型数据库驱动 | **asyncpg / aiomysql / oracledb / pyhive** | 原生异步驱动适配器模式，支持 PostgreSQL / MySQL / Oracle / Hive 四类数据源 |
| 向量数据库驱动（v0.5） | **pymilvus**（懒导入） | Milvus 向量库，ANN 检索、过滤表达式 |
| 图数据库驱动（v0.5） | **neo4j**（懒导入） | Neo4j 异步 driver、Cypher 查询、连接池 |
| 多模态处理（v0.5） | **Pillow / ffmpeg / pydub**（懒导入） | 图片 OCR、视频关键帧、音频 ASR |
| HTTP 客户端 | **httpx** | 异步 API 调用、连接池、流式响应 |
| 任务队列 | **内存 / Redis** | 双模式可切换 |
| 分布式锁（v0.5） | **Redis SET NX + Lua** | 防误解锁的 token 机制；单进程降级为 asyncio.Lock |
| 配置管理 | **Pydantic Settings** | 环境变量 / YAML 双源 |
| 日志 | **标准库 `logging`** | 结构化、零依赖、异步友好 |
| 可观测性（v0.6） | **Prometheus + OpenTelemetry**（懒导入） | 自定义指标、分布式追踪、`/metrics` 端点 |
| 对象存储（v0.6） | **minio**（懒导入） | S3 兼容对象存储、预签名 URL、分片上传 |
| 配置热加载（v0.6） | **watchdog**（懒导入，可选） | 文件监听；未安装时自动降级为 polling |
| 消息队列（v0.6） | **aiokafka / aio-pika**（懒导入） | Kafka / RabbitMQ 触发器，事件驱动工作流 |

---

## 快速开始

### 1. 安装

```bash
pip install -r requirements.txt
```

### 2. 配置文件

创建 `config/models.yaml`：

```yaml
models:
  gpt-4o:
    model_id: gpt-4o
    model_name: gpt-4o
    api_base: https://api.openai.com/v1
    api_key: ${OPENAI_API_KEY}
    max_tokens: 4096
    tags: [capable, fast]
    cost_per_1k_input: 0.0025
    cost_per_1k_output: 0.01
  deepseek:
    model_id: deepseek
    model_name: deepseek-chat
    api_base: https://api.deepseek.com/v1
    api_key: ${DEEPSEEK_API_KEY}
    tags: [cheap, capable]
    cost_per_1k_input: 0.00014
    cost_per_1k_output: 0.00028

routing_rules:
  - task_type: summary
    preferred_tags: [cheap]
    max_cost: 0.001
  - preferred_tags: [capable]
    fallback_model_id: gpt-4o
```

### 3. 编写工作流

```python
from icore.core.base_task import BaseTask
from icore.core.models import BaseTaskInput, BaseTaskOutput
from icore.core.registry import register_task
from icore.core.task_context import TaskContext
from icore.engine.base_workflow import BaseWorkflow
from icore.engine.dag import DAG

# 1. 定义任务输入类（每个 Task 都有自己的输入类）
class SummaryInput(BaseTaskInput):
    text: str

# 2. 实现具体 Task
@register_task("simple_summary")
class SimpleSummaryTask(BaseTask):
    name = "simple_summary"
    description = "Summarize input text"
    input_model = SummaryInput

    async def prepare(self, ctx: TaskContext) -> None:
        self._model = ctx.get_model_adapter()

    async def execute(self, ctx: TaskContext, inp: SummaryInput) -> BaseTaskOutput:
        resp = await self._model.chat([
            {"role": "user", "content": f"Summarize: {inp.text}"}
        ])
        return BaseTaskOutput.success(summary=resp.get("content", ""))

# 3. 定义 Workflow
class SimpleWorkflow(BaseWorkflow):
    name = "simple_workflow"
    description = "Simple single-task summary workflow"

    def define(self) -> DAG:
        dag = DAG()
        dag.add_node("summarize", self.get_task("simple_summary"))
        return dag
```

### 4. 启动服务

```bash
# 开发模式
uvicorn icore.api.main:app --host 0.0.0.0 --port 8000 --reload

# 生产模式
uvicorn icore.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 4a. Docker Compose 一键部署（v0.6）

```bash
# 仅 icore API（轻量）
docker-compose -f deploy/docker-compose.yaml up -d

# 全栈：icore + PostgreSQL + Redis + Milvus + Neo4j + MinIO（推荐）
docker-compose -f deploy/docker-compose.full.yaml up -d

# 叠加可观测性栈：Prometheus + Grafana + Jaeger + Loki
docker-compose -f deploy/docker-compose.full.yaml \
               -f deploy/docker-compose.obs.yaml up -d
```

启动后约 30 秒内可用：

| 服务 | 地址 | 用途 |
|------|------|------|
| icore API | http://localhost:8000/docs | Swagger UI、`/invoke`、`/health`、`/metrics` |
| PostgreSQL | localhost:5432 | 持久化、审计日志、断点续跑 |
| Redis | localhost:6379 | 分布式锁、任务队列、幂等缓存 |
| Milvus | localhost:19530 | 向量检索（Attu UI: http://localhost:8001）|
| Neo4j | http://localhost:7474 | 图谱（bolt://localhost:7687）|
| MinIO | http://localhost:9001 | 对象存储控制台（S3 API: localhost:9000）|
| Prometheus | http://localhost:9090 | 指标采集 |
| Grafana | http://localhost:3000 | 指标看板（admin/admin）|
| Jaeger | http://localhost:16686 | 分布式追踪 |

### 5. 调用接口

```bash
# 健康检查
curl http://localhost:8000/health

# 主调用（非流式）
curl -X POST http://localhost:8000/invoke \
  -H "Content-Type: application/json" \
  -d '{
    "workflow_name": "simple_workflow",
    "params": {"text": "Long article text here..."},
    "task_id": "my-instance-001",
    "stream": false
  }'

# 流式调用
curl -X POST http://localhost:8000/invoke \
  -H "Content-Type: application/json" \
  -d '{
    "workflow_name": "simple_workflow",
    "params": {"text": "Long article text here..."},
    "task_id": "my-instance-002",
    "stream": true
  }'

# 带回调
curl -X POST http://localhost:8000/invoke \
  -H "Content-Type: application/json" \
  -d '{
    "workflow_name": "simple_workflow",
    "params": {"text": "Text"},
    "task_id": "my-instance-003",
    "callback_url": "https://my-app.com/webhook",
    "model_id": "gpt-4o"
  }'
```

---

## 配置指南

### 环境变量

所有配置可通过 `ICORE_` 前缀的环境变量设置：

```bash
export ICORE_API_PORT=8000
export ICORE_CONCURRENCY_MAX_CONCURRENT_TASKS=200
export ICORE_LOG_LEVEL=DEBUG
```

### 数据库配置

`config/databases.yaml`：

```yaml
connections:
  main_db:
    db_type: postgresql
    host: localhost
    port: 5432
    username: user
    password: ${DB_PASSWORD}
    database: mydb
    pool_size: 10
  analytics_db:
    db_type: oracle
    host: 10.0.0.5
    port: 1521
    username: analyst
    password: ${ORACLE_PASSWORD}
    database: ORCLPDB1
```

### API 参数说明

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `workflow_name` | string | 是 | 已注册的工作流名称 |
| `params` | dict | 是 | 工作流输入参数（匹配 workflow 的 input_model） |
| `task_id` | string | 否 | 任务实例 ID（用于去重与追踪，不传则自动生成） |
| `callback_url` | string | 否 | 异步结果回调地址 |
| `model_id` | string | 否 | 指定模型 ID（不指定则自动路由） |
| `stream` | bool | 否 | 是否 SSE 流式返回（默认 false） |
| `metadata` | dict | 否 | 扩展元数据 |
| `idempotency_key` | string | 否 | 幂等键（v0.5）：相同 key 的重复请求返回原始结果，不重新执行，TTL 24 小时 |

---

## 架构概览

icore 采用分层架构，严格单向依赖：

```
api / services ──-> engine ──-> core
                    │  │  │
                    │  │  ├──-> db           (SQL)
                    │  │  ├──-> models        (LLM + 多模态)
                    │  │  ├──-> vectorstore   (Milvus)
                    │  │  ├──-> graphstore    (Neo4j)
                    │  │  ├──-> media         (图片/视频/音频)
                    │  │  └──-> objectstore  (MinIO / S3)  [v0.6]
                    │  │
                    └──-> lock / circuit_breaker / idempotency
                          auth / cache / prompts / retrieval  [v0.6]
                          observability / persistence / triggers  [v0.6]
                          security  [v0.6]
```

- **core**：最底层的任务抽象（`BaseTask`、`BaseTaskInput`、`TaskContext`、`TaskRegistry`）
- **db**：数据库连接层（`BaseConnector`、`ConnectionPool`、`DBManager` + 4 个适配器）
- **models**：LLM 模型管理层（`BaseModelAdapter`、`ModelManager`、`ModelRouter`、`VisionModelAdapter`）
- **vectorstore**（v0.5）：向量存储层（`BaseVectorStore` + `MilvusAdapter` / `InMemoryVectorStore`）
- **graphstore**（v0.5）：图存储层（`BaseGraphStore` + `Neo4jAdapter` / `InMemoryGraphStore`）
- **media**（v0.5）：多模态文件处理（`MediaFile` + `ImageProcessor` / `VideoProcessor` / `AudioProcessor`）
- **objectstore**（v0.6）：对象存储抽象（`BaseObjectStore` + `MinIOAdapter` / `InMemoryObjectStore`）
- **auth**（v0.6）：鉴权层（API Key / JWT 双模式 + RBAC）
- **cache**（v0.6）：LLM 响应缓存（`SemanticCache` L1 精确 + L2 向量相似）
- **prompts**（v0.6）：Prompt 模板管理（YAML 加载、版本切换、A/B 测试）
- **retrieval**（v0.6）：混合检索（`HybridRetriever` Dense + BM25 + RRF + Reranker）
- **observability**（v0.6）：可观测性（Prometheus metrics、OpenTelemetry tracing、结构化日志）
- **persistence**（v0.6）：工作流持久化（`workflow_executions` / `task_executions` 表 + 断点续跑）
- **triggers**（v0.6）：消息队列触发器（Kafka / RabbitMQ -> 工作流）
- **security**（v0.6）：安全加固（Prompt 注入检测 + PII 脱敏）
- **engine**：工作流引擎（`BaseWorkflow`、`DAG`、`WorkflowExecutor`、任务队列、并发控制、实例管理、熔断器、分布式锁、Agent 协作框架、退避策略、死信队列、Saga 补偿、全链路背压、配置热加载、优雅降级）
- **api**：HTTP 接口层（FastAPI 2 端点、Pydantic Schema、回调、SSE 流、幂等键缓存、全局异常中间件、鉴权依赖、`/metrics` 端点）
- **services**：服务暴露层（MCP、Tool 服务、Streamlit、SSE 适配器）
- **workflows**：具体业务工作流实现（示例：文档摘要、实体提取、周报生成、RAG 问答、知识图谱、多模态、Agent 协作、Saga 订单处理、对象存储报表导出）

完整架构请参阅：[docs/01-architecture-overview.md](docs/01-architecture-overview.md)

---

## 示例工作流

项目内含九个开箱即用的示例，覆盖 v0.4 基础场景、v0.5 高级场景与 v0.6 工程化场景：

| 工作流 | 文件 | 说明 |
|--------|------|------|
| 大文档摘要 | `icore/workflows/examples/document_summary.py` | 分块 -> 逐块摘要 -> 合并，多 Task 的 DAG |
| 实体关系提取 | `icore/workflows/examples/entity_extraction.py` | 提取 -> 标准化 -> 格式化输出 |
| 周报生成 | `icore/workflows/examples/weekly_report.py` | DB 查询 -> LLM 生成 -> 格式化，演示数据库集成 |
| RAG 问答（v0.5） | `icore/workflows/examples/rag_qa.py` | embed 查询 -> 向量检索 top-K -> LLM 基于上下文生成答案 |
| 知识图谱（v0.5） | `icore/workflows/examples/knowledge_graph.py` | 实体抽取 -> 写入图库（带分布式锁）-> 子图查询 + GraphRAG 答案 |
| 图片描述（v0.5） | `icore/workflows/examples/multimodal.py` | Vision 模型描述图片；OCR 提取文本 -> LLM 摘要 |
| Agent 协作（v0.6） | `icore/workflows/examples/agent_demo.py` | REACT / SUPERVISOR / SWARM 三种 Agent 协作模式演示 |
| Saga 订单处理（v0.6） | `icore/workflows/examples/order_processing.py` | 扣库存 -> 支付 -> 通知，失败自动补偿回滚 |
| 对象存储报表导出（v0.6） | `icore/workflows/examples/report_export.py` | 分析 -> 渲染 -> 上传 MinIO -> 返回预签名 URL |

---

## 设计文档

完整设计文档位于 `docs/` 目录：

| 编号 | 文档 | 内容 |
|------|------|------|
| 01 | [系统架构总览](docs/01-architecture-overview.md) | 架构、技术栈、模块拆解、数据流 |
| 02 | [任务抽象层](docs/02-task-abstraction.md) | BaseTask、输入输出模型、TaskContext、注册表 |
| 03 | [数据库连接层](docs/03-database-layer.md) | BaseConnector、连接池、多数据源适配器 |
| 04 | [工作流引擎](docs/04-workflow-engine.md) | DAG、BaseWorkflow、调度执行、状态管理 |
| 05 | [模型管理与路由](docs/05-model-management.md) | 多模型配置、自动路由、故障转移 |
| 06 | [API 层](docs/06-api-layer.md) | 2 端点设计、Swagger、SSE、回调 |
| 07 | [并发与性能](docs/07-concurrency.md) | asyncio 模型、任务队列、并发控制、背压 |
| 08 | [服务暴露层](docs/08-service-exposure.md) | MCP、Tool、Streamlit、SSE 适配器 |
| 09 | [示例工作流](docs/09-examples.md) | 三个完整示例的开发指南 |
| 10 | [目录结构](docs/10-directory-structure.md) | 完整项目树与文件说明 |
| 11 | [v0.5 增强设计](docs/11-v0.5-enhancement.md) | 向量库 / 图谱 / 多模态 / 健壮性增强方案 |
| 12 | [v0.6 增强设计](docs/12-v0.6-enhancement.md) | 可观测性 / Agent / 缓存 / 持久化 / 鉴权 / 混合检索 / 对象存储 / 弹性 / 部署 |
| 13 | [v0.6 实施总结](docs/13-v0.6-implementation.md) | v0.6 落地实施记录、模块清单、测试矩阵 |

---

## 项目统计

- **Python 源文件**：100+ 个（v0.6 新增 objectstore / auth / cache / prompts / retrieval / observability / persistence / triggers / security / agent / backoff / dead_letter_queue / saga / backpressure / hot_reload / graceful_degradation 等模块）
- **设计文档**：13 篇（含 80+ 张 Mermaid 图）
- **代码总量**：~35,000 行
- **核心抽象基类**：15 个（BaseTask、BaseWorkflow、BaseConnector、BaseModelAdapter、BaseServiceExposer、BaseVectorStore、BaseGraphStore、BaseMediaProcessor、BaseDistributedLock、BaseObjectStore、BaseReranker、BaseDLQBackend、BaseAuthenticator、BaseInjectionDetector、BasePIIDetector）
- **数据库适配器**：4 个（PostgreSQL、Oracle、Hive、MySQL）
- **向量库适配器**：2 个（Milvus、InMemory）
- **图库适配器**：2 个（Neo4j、InMemory）
- **对象存储适配器**：2 个（MinIO、InMemory）
- **媒体处理器**：3 个（Image、Video、Audio）
- **Agent 协作模式**：3 种（REACT、SUPERVISOR、SWARM）
- **退避策略**：4 种（CONSTANT、LINEAR、EXPONENTIAL、EXPONENTIAL_JITTER）
- **示例工作流**：9 个（含 v0.6 Agent 协作、Saga 订单、对象存储报表）
- **测试用例**：1121 passed + 2 skipped

---

## 许可证

本项目为内部设计文档资产，未授权外部发布。

---

*icore — 让大模型工作流，用代码定义。*
