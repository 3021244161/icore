# AGENTS.md - icore 开发规范与系统须知

> 本文档面向 AI 协作代理（含人类协作者），定义 icore 项目的开发规范、协作约定、技术选型与质量标准。任何对本项目进行修改的 Agent（Claude / Cursor / TRAE / Codex 等）在动手前**必须**先读完本文档。

---

## 0. 项目定位

**icore** 是一个企业级 LLM 工作流编排平台，通过编码形式（非拖拽）将大模型任务编排成可复用、可调度的工作流。

- **核心范式**：Coding-Driven Workflow Orchestration
- **关键能力**：DAG 工作流引擎 / 多模型管理 / 多数据库支持 / 统一 API（2 端点）/ SSE 流式 / 回调 / 并发治理 / 多协议服务暴露
- **交付状态**：可运行、可测试、可部署；10 份设计文档 + 全量代码 + 测试套件

---

## 1. 协作风格（来自用户长期偏好）

| 维度 | 偏好 |
|------|------|
| 沟通语言 | 中文 |
| 行动风格 | 大刀阔斧、直接执行；不喜欢繁琐确认 |
| 分析彻底性 | 偏好全量数据而非截断数据；重构要彻底 |
| 进度反馈 | 长任务需提供进度反馈（TodoWrite / 进度条） |
| 网络故障 | Docker 重建 / GitHub 连接 / 外网访问失败一次后，立即中断并把命令交还用户手动执行，不要反复重试 |
| 容器改动 | ZhiMou/icore 后端或容器相关代码改动完成后，顺手重建 Docker 容器（同样遵循网络故障协作规则） |
| 技术偏好 | 倾向 LangGraph 构建复杂工作流；关注并发控制和执行效率 |

---

## 2. 技术栈（与代码实际一致，文档与代码必须同步）

| 层面 | 技术 | 说明 |
|------|------|------|
| Web 框架 | FastAPI | 原生异步、自动 Swagger、Pydantic 集成 |
| 异步运行时 | asyncio + uvicorn | 高并发 I/O 密集 |
| 数据校验 | Pydantic v2 | 类型安全、Schema 生成 |
| **数据库驱动** | **asyncpg / aiomysql / oracledb / pyhive** | **原生异步驱动 + 适配器模式，不用 ORM** |
| HTTP 客户端 | httpx | 异步、连接池、流式 |
| 任务队列 | 内存 / Redis | 双模式可切换 |
| 配置管理 | Pydantic Settings | 环境变量 + YAML 双源 |
| **日志** | **标准库 `logging`** | **零依赖、与 asyncio 兼容；不用 loguru** |
| 容器化 | Docker + docker-compose | |

> ⚠️ **重要**：项目**不使用** SQLAlchemy / loguru。历史文档曾误声明，已在 2026-08-01 修正。任何新文档、注释、README 都不得再引入这两项声明。

---

## 3. 项目结构与分层依赖

```
icore/
├── core/        # 最底层任务抽象（BaseTask / TaskContext / Registry / Models）
├── db/          # 数据库连接层（BaseConnector / ConnectionPool / DBManager + 4 适配器）
├── models/      # LLM 模型管理（BaseModelAdapter / ModelManager / ModelRouter）
├── engine/      # 工作流引擎（BaseWorkflow / DAG / Executor / 并发控制 / 实例管理 / 任务队列）
├── api/         # HTTP 接口层（FastAPI 2 端点 / Schema / 回调 / SSE 流）
├── services/    # 服务暴露层（MCP / Tool / Streamlit / SSE 适配器）
└── workflows/   # 具体业务工作流（examples/ 下三个开箱示例）
```

**严格单向依赖**：

```
api / services ──→ engine ──→ core
                    │  │
                    ├──→ db
                    └──→ models
```

- `core` 是纯抽象层，**零外部依赖**（仅 Pydantic + 标准库）
- 禁止反向依赖、禁止跨层越级依赖
- `db` 与 `models` 之间互不依赖

---

## 4. 关键设计原则（不得违背）

### 4.1 依赖注入走官方 setter

`TaskContext` 是 Pydantic frozen model，注入 ModelManager / DBManager **必须**用：

```python
ctx.set_model_manager(mgr)
ctx.set_db_manager(mgr)
```

**禁止**用 `object.__setattr__(ctx, "_model_manager", ...)` 这种绕过 setter 的 hack（除非你是 `TaskContext` 自身实现 setter，那是唯一例外）。已有回归测试 `tests/test_services.py::TestNoSetattrHack` 守护此规则。

### 4.2 并发治理必须接线

`ConcurrencyController` / `TaskQueue` / `TaskInstanceManager` 三个组件**已在 API 层接线**：

- `bootstrap.create_production_app()` 负责构建并注入
- `/invoke` 端点执行流：背压检查（503）→ 实例登记（去重）→ `acquire()` 槽位 → 状态流转（PENDING→RUNNING→COMPLETED/FAILED/CANCELLED）
- SSE 流式路径由 `SSEStreamHandler` 驱动状态流转
- 任何新增的执行入口（如未来的 gRPC 端点）都必须套用同样的治理模式

### 4.3 `_ConcurrencySlot` 信号量安全

`acquire()` 先取 global semaphore 再取 per-workflow semaphore。若第二次 acquire 失败（含 CancelledError），**必须**释放已获取的 global semaphore。已有 try/finally 守护，不要回退。

### 4.4 `get_terminal_output` 返回结构不对称（已知设计）

- **单终端节点**：返回该节点原始 `BaseTaskOutput`，`output.data` 是 task 自身数据，**无 `results` 包装**
- **多终端节点**：返回新建的 `BaseTaskOutput`，`output.data` 形如 `{"results": {node_id: data, ...}}`
- 调用方需按 `len(terminal_nodes)` 或 `data` 是否含 `results` 键分支处理

详见 [icore/engine/executor.py](icore/engine/executor.py) 的 `get_terminal_output` docstring 与 [docs/04-workflow-engine.md](docs/04-workflow-engine.md) §6.6。

### 4.5 适配器模式贯穿始终

- 数据库：`BaseConnector` + PostgreSQL/MySQL/Oracle/Hive 4 适配器，驱动**懒导入**（模块可在驱动未装时 import）
- 模型：`BaseModelAdapter` + OpenAICompatibleAdapter
- 服务：`BaseServiceExposer` + MCP/Tool/Streamlit/SSE 4 exposer

新增类型时优先扩展现有抽象，不要另起炉灶。

---

## 5. API 设计约束

- **恰好两个端点**：`GET /health` + `POST /invoke`，不得随意新增
- 异常处理统一走 `@app.exception_handler`：KeyError→404、ValueError→422、BackpressureError→503、其他→500
- 所有响应经 `app.state._bg_tasks` 持有强引用，防止后台任务被 GC 中断

---

## 6. 测试规范

### 6.1 测试覆盖要求

| 模块 | 测试文件 | 状态 |
|------|----------|------|
| core (task/context/registry) | test_registry_and_task.py | ✅ |
| engine (dag/executor) | test_dag.py / test_executor.py | ✅ |
| api | test_api.py | ✅ |
| config + bootstrap | test_config_bootstrap.py | ✅ |
| 示例工作流 | test_example_workflows.py | ✅ |
| **并发组件** | test_concurrency.py | ✅ |
| **db 适配器 + 连接池 + DBManager** | test_db.py | ✅ |
| **services 4 个 exposer** | test_services.py | ✅ |

**任何新增模块必须配套测试**。PR 不带测试不予合并。

### 6.2 测试约定

- 使用 `pytest` + `pytest-asyncio`（`asyncio_mode = "auto"`）
- 离线测试：`FakeModelAdapter` 模拟 LLM，`_FakeConnector` 模拟 DB，**不**连真实网络
- 测试文件命名：`tests/test_<模块名>.py`
- 每个 public 方法至少 1 个 happy path + 1 个 error path
- 回归测试用 `TestNo...` 命名类明确意图

### 6.3 运行测试

```bash
# 全量
python -m pytest tests -q

# 详细
python -m pytest tests -v --tb=short

# 单文件
python -m pytest tests/test_concurrency.py -v
```

当前基线：**279 passed, 1 skipped**（skip 是因为环境已装 asyncpg 无法测 ImportError 路径）。

---

## 7. 依赖管理

- `requirements.txt`：**运行时 + 测试时**依赖（pytest / pytest-asyncio 已含），`pip install -r requirements.txt` 即可跑测试
- `pyproject.toml` 的 `[dev]` extra：额外开发工具（pytest-cov 等）
- 数据库驱动按需安装，未装时模块仍可 import（懒导入设计）

---

## 8. Git 提交规范

### 8.1 Conventional Commits

```
<type>[scope]: <description>

[optional body]
```

| Type | 用途 |
|------|------|
| feat | 新功能 |
| fix | 修复 bug |
| docs | 仅文档 |
| refactor | 重构（无功能/修复） |
| test | 测试 |
| build | 构建/依赖 |
| chore | 杂项维护 |

### 8.2 提交原则

- **一次提交一个逻辑变更**
- 描述用现在时祈使句："add" 而非 "added"
- 描述 < 72 字符
- **禁止** `git add -A` / `git add .`（避免误提交 secrets / 大文件）
- **禁止**提交 `.env` / `credentials.json` / 私钥
- **禁止** `--force` 到 main / master
- 临时输出文件（`pytest_out*.txt`）已加入 `.gitignore`，不要提交

### 8.3 安全协议

- 不修改 git config
- 不执行 `--force` / `reset --hard` / `checkout .` 等破坏性操作，除非用户明确要求
- 不跳过 hooks（`--no-verify`），除非用户要求
- hook 失败时修复后新建 commit，不 amend

---

## 9. 文档同步责任

文档与代码不一致是技术债。修改代码时**同步**修改对应文档：

| 改动位置 | 必须同步的文档 |
|----------|----------------|
| 技术栈 | README.md + docs/01-architecture-overview.md §3 |
| 目录结构 | docs/10-directory-structure.md |
| 新增模块 | docs/ 对应编号文档 + README.md 架构概览 |
| API 端点 | docs/06-api-layer.md |
| 并发行为 | docs/07-concurrency.md |
| `get_terminal_output` 行为 | docs/04 §6.6 + executor.py docstring |

---

## 10. 常见陷阱与经验教训

1. **`object.__setattr__` 反模式**：曾出现在 4 个 service exposer，已全部修复并用回归测试守护。新 exposer 必须用 setter。
2. **并发组件"写了没接线"**：曾出现 ConcurrencyController 实现完整但 API 层未调用。新增治理组件必须同步接入 bootstrap + API。
3. **文档声明与代码不符**：曾误声明 SQLAlchemy / loguru。验收必查文档-代码一致性。
4. **信号量泄漏**：两次 acquire 之间若被取消，第一次的 semaphore 必须释放。
5. **pytest-asyncio 缺失**：曾经因 `pyproject.toml` 配了 `asyncio_mode=auto` 但没装插件导致大量假失败。已在 requirements.txt 补齐。
6. **多终端返回结构不对称**：`get_terminal_output` 单/多终端返回形态不同，调用方必须分支处理。

---

## 11. Agent 行为检查清单

动手前自问：

- [ ] 我读过 AGENTS.md 了吗？
- [ ] 我了解目标模块所在的分层和依赖方向吗？
- [ ] 我知道相关的 setter / 接线点在哪吗？
- [ ] 我的改动会破坏文档-代码一致性吗？需要同步改文档吗？
- [ ] 我加了测试吗？
- [ ] 我用了 `object.__setattr__` 吗？（如果用了，停下来换 setter）
- [ ] 我新增了 API 端点吗？（如果新增了，停下来——本项目只允许 2 个端点）
- [ ] 我引入了 SQLAlchemy / loguru 吗？（如果引入了，停下来——本项目不用）
- [ ] 长任务我给进度反馈了吗？
- [ ] 网络操作失败一次后，我是不是该把命令交给用户？

---

*本文档由 icore 项目维护，最后更新：2026-08-01。*
