# icore — 企业级 LLM 工作流编排平台

**Coding-Driven Workflow Orchestration Platform**

icore 是一个企业级工作流能力平台，通过编码形式将大模型（LLM）驱动的任务编排成可复用、可调度的工作流。不提供拖拽式的可视化编辑器——工作流完全通过 Python 代码定义，充分利用编程语言的表达能力（条件分支、循环、异常处理、子流程嵌套）实现复杂业务逻辑。

---

## 核心特性

- **DAG 工作流引擎**：基于有向无环图的任务调度，支持拓扑排序、并行执行、子工作流嵌套（工作流调度工作流）
- **任务抽象层**：`BaseTask` 抽象基类，每个 Task 有强类型的输入/输出 Pydantic 模型，`TaskRegistry` 注册与管理
- **多模型管理**：配置多个 OpenAI 兼容模型（GPT-4o、Claude、DeepSeek、本地模型等），自动路由 + 故障转移 + 成本感知选择
- **多数据库支持**：统一 `BaseConnector` 抽象，适配器模式支持 PostgreSQL、Oracle、Hive、MySQL 四类数据源，异步连接池
- **统一 API**：仅两个端点——`GET /health`（系统健康检查）和 `POST /invoke`（主调用接口），Swagger 自动生成
- **流式返回**：支持 `stream=true` 的 SSE（Server-Sent Events）流式输出，适合 LLM 逐 token 生成场景
- **回调机制**：异步结果投递到 `callback_url`，解耦请求方
- **高并发处理**：全 asyncio 架构、内存/Redis 双模式任务队列、Token Bucket 限流、按工作流/任务类型的并发控制、背压机制
- **服务暴露**：工作流可封装为 MCP 服务、Tool 服务（函数调用）、Streamlit 应用、SSE 端点——一套工作流多协议消费
- **依赖注入**：`TaskContext` 注入模型适配器、数据库连接、元数据，Task 不硬编码依赖，可测试、可替换

---

## 技术栈

| 层面 | 技术 | 用途 |
|------|------|------|
| Web 框架 | **FastAPI** | 原生异步、自动 Swagger、Pydantic 集成 |
| 异步运行时 | **asyncio + uvicorn** | 高并发 I/O 密集场景 |
| 数据校验 | **Pydantic v2** | 类型安全、自动校验、Schema 生成 |
| 数据库驱动 | **asyncpg / aiomysql / oracledb / pyhive** | 原生异步驱动适配器模式，支持 PostgreSQL / MySQL / Oracle / Hive 四类数据源 |
| HTTP 客户端 | **httpx** | 异步 API 调用、连接池、流式响应 |
| 任务队列 | **内存 / Redis** | 双模式可切换 |
| 配置管理 | **Pydantic Settings** | 环境变量 / YAML 双源 |
| 日志 | **标准库 `logging`** | 结构化、零依赖、异步友好 |

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

---

## 架构概览

icore 采用分层架构，严格单向依赖：

```
api / services ──→ engine ──→ core
                    │  │
                    ├──→ db
                    └──→ models
```

- **core**：最底层的任务抽象（`BaseTask`、`BaseTaskInput`、`TaskContext`、`TaskRegistry`）
- **db**：数据库连接层（`BaseConnector`、`ConnectionPool`、`DBManager` + 4 个适配器）
- **models**：LLM 模型管理层（`BaseModelAdapter`、`ModelManager`、`ModelRouter`）
- **engine**：工作流引擎（`BaseWorkflow`、`DAG`、`WorkflowExecutor`、任务队列、并发控制、实例管理）
- **api**：HTTP 接口层（FastAPI 2 端点、Pydantic Schema、回调、SSE 流）
- **services**：服务暴露层（MCP、Tool 服务、Streamlit、SSE 适配器）
- **workflows**：具体业务工作流实现（示例：文档摘要、实体提取、周报生成）

完整架构请参阅：[docs/01-architecture-overview.md](docs/01-architecture-overview.md)

---

## 示例工作流

项目内含三个开箱即用的示例：

| 工作流 | 文件 | 说明 |
|--------|------|------|
| 大文档摘要 | `icore/workflows/examples/document_summary.py` | 分块 → 逐块摘要 → 合并，2 个 Task 的 DAG |
| 实体关系提取 | `icore/workflows/examples/entity_extraction.py` | 提取 → 标准化 → 格式化输出 |
| 周报生成 | `icore/workflows/examples/weekly_report.py` | DB 查询 → LLM 生成 → 格式化，演示数据库集成 |

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

---

## 项目统计

- **Python 源文件**：44 个
- **设计文档**：10 篇（含 50+ 张 Mermaid 图）
- **代码总量**：~13,000 行
- **核心抽象基类**：5 个（BaseTask、BaseWorkflow、BaseConnector、BaseModelAdapter、BaseServiceExposer）
- **数据库适配器**：4 个（PostgreSQL、Oracle、Hive、MySQL）

---

## 许可证

本项目为内部设计文档资产，未授权外部发布。

---

*icore — 让大模型工作流，用代码定义。*
