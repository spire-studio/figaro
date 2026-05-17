<div align="center">
  <h1>🎼 Figaro</h1>
  <p><em>一个由 AI 驱动实验编排的智能联邦学习平台</em></p>
  <p>
    <a href="https://github.com/spire-studio/figaro/actions/workflows/ci-backend.yml"><img src="https://github.com/spire-studio/figaro/actions/workflows/ci-backend.yml/badge.svg" alt="CI Backend"></a>
    <a href="https://github.com/spire-studio/figaro/actions/workflows/ci-frontend.yml"><img src="https://github.com/spire-studio/figaro/actions/workflows/ci-frontend.yml/badge.svg" alt="CI Frontend"></a>
    <img src="https://img.shields.io/badge/python-%E2%89%A53.11-blue?logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white" alt="FastAPI">
    <img src="https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=white" alt="React">
    <img src="https://img.shields.io/badge/PyTorch-2.5-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
    <img src="https://img.shields.io/badge/LangGraph-agent-1C3C3C?logo=langchain&logoColor=white" alt="LangGraph">
    <a href="https://github.com/spire-studio/figaro/stargazers"><img src="https://img.shields.io/github/stars/spire-studio/figaro?style=social" alt="Stars"></a>
  </p>
  <p>
    <a href="./README.md">English</a> | <strong>简体中文</strong>
  </p>
</div>

---

🎼 **Figaro** 把自然语言实验描述直接变成可复现的联邦学习实验。你只需要用一句话说明要对比什么，Agent 会自动展开、调度并汇总结果。

⚡ 基于 **FastAPI + LangGraph + PyTorch** 构建，前端使用 React 实时展示训练进度与对比视图。

## 📢 动态

- **2026-04-14** 🚦 CI 上线 —— GitHub Actions 现在会在每个 PR 上跑后端 pytest 和前端 `tsc + vite build`。
- **2026-04-14** 🧹 重要重构：移除遗留的 attack/defense 模块，项目更名为 **Figaro**，强化可复现性并清理日志处理。
- **2026-04-10** 🎉 **Figaro v0** —— 基于 LangGraph 的实验 Agent 首次公开发布。

## ✨ 核心特性

🤖 **智能实验 Agent** —— 用自然语言描述实验，基于 LangGraph 的 Agent 会自动规划、展开、启动并汇总结果。

🖥️ **单机仿真模式** —— 单机联邦学习，支持 CIFAR-10 / MNIST、Dirichlet 非 IID 划分、CNN / ResNet、FedAvg、CKKS 同态加密和 Top-K 稀疏化。

🌐 **分布式模式** —— 基于 gRPC 的多节点训练，Server / Worker 架构，Docker 一键部署。

📊 **全栈实现** —— FastAPI + PostgreSQL + SQLAlchemy 后端，React + Vite + TypeScript 前端，Agent Tab 提供实验输入、实时进度和对比结果。

🔁 **可复现优先** —— 配置驱动、固定随机种子、产物持久化，每个实验都能重新跑一遍。

## 🎯 Agent 能做什么

```
"Compare CIFAR-10 non-IID with alpha=0.1, 0.3, 0.5"
→ Agent 展开为 3 个实验 → 依次运行 → 输出对比表格
```

```
"Compare 10 clients vs 20 clients with 5/10/20 rounds"
→ Agent 展开为 2×3 = 6 个实验 → 笛卡尔积 → 矩阵视图
```

## 🏗️ 架构

```
┌──────────────────────────────────────────────────────────────┐
│                   React + Vite 前端                          │
│              (Agent Tab · 进度 · 对比视图)                   │
└───────────────────────────┬──────────────────────────────────┘
                            │ REST / SSE
┌───────────────────────────▼──────────────────────────────────┐
│                     FastAPI 后端                             │
│   ┌────────────┐  ┌──────────────┐  ┌──────────────────┐     │
│   │ LangGraph  │→ │  Experiment  │→ │   Run Service    │     │
│   │   Agent    │  │   Service    │  │  (sim / dist)    │     │
│   └────────────┘  └──────────────┘  └────────┬─────────┘     │
│           ▲                                  │               │
│           │              PostgreSQL          ▼               │
│   ┌───────┴────────┐   ┌────────────┐   ┌──────────────┐     │
│   │  LLM Service   │   │ Job / Run  │   │ fl_core libs │     │
│   │ (OpenAI 等)    │   │ Repository │   │ FedAvg · CKKS│     │
│   └────────────────┘   └────────────┘   └──────────────┘     │
└──────────────────────────────────────────────────────────────┘
```

## 📚 目录

- [动态](#-动态)
- [核心特性](#-核心特性)
- [架构](#️-架构)
- [安装](#️-安装)
- [快速开始](#-快速开始)
- [使用 Agent](#-使用-agent)
- [配置](#️-配置)
- [项目结构](#-项目结构)
- [贡献 & Roadmap](#-贡献--roadmap)

## ⚙️ 安装

```bash
git clone https://github.com/spire-studio/figaro.git
cd figaro
uv sync

cp .env.example .env
# 编辑 .env，设置：
#   OPENAI_API_KEY=your-api-key
#   POSTGRES_PASSWORD=postgres
```

## 🚀 快速开始

### 方式 A：Docker 一键启动

```bash
./scripts/docker-dev-up.sh
```

### 方式 B：手动启动（推荐 GPU 开发）

**第 1 步 — PostgreSQL**（Docker 或本地）：
```bash
docker run -d --name figaro-pg -p 5433:5432 \
  -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=figaro postgres:16
```

**第 2 步 — 后端**：
```bash
POSTGRES_HOST=localhost POSTGRES_PORT=5433 POSTGRES_PASSWORD=postgres POSTGRES_DB=figaro \
  PYTHONPATH=libs:apps/backend/runners \
  uv run uvicorn app.main:app --app-dir apps/backend --host 0.0.0.0 --port 8000 --reload
```

指定 GPU（例如 GPU 1）：
```bash
CUDA_VISIBLE_DEVICES=1 POSTGRES_HOST=localhost ... uv run uvicorn ...
```

**第 3 步 — 前端**（在另一个终端）：
```bash
cd apps/frontend && pnpm install && pnpm dev
```

**访问**：
- 前端：`http://localhost:5173`
- 后端 API：`http://localhost:8000/docs`

### 分布式部署

```bash
./scripts/docker-dist-up.sh
```

## 🤖 使用 Agent

在前端打开 **Agent** Tab，用自然语言描述你的实验：

- "Compare CIFAR-10 non-IID with alpha=0.1, 0.3, 0.5"
- "Compare 10 clients vs 20 clients with FedAvg"
- "Test training rounds 10, 20, 50 on accuracy"

Agent 会解析你的请求，展开为一组具体实验，依次运行并生成对比报告。

## 🛠️ 配置

```yaml
dataset:
  name: "CIFAR-10"
  distribution: "non_iid"
  alpha: 0.5

model:
  name: "CNN"
  input_shape: [32, 32, 3]
  num_classes: 10

federated:
  num_clients: 10
  num_rounds: 20
  clients_per_round: 5
  local_epochs: 5
  learning_rate: 0.01
  aggregation: "fedavg"
```

## 📁 项目结构

```
figaro/
├── apps/
│   ├── backend/
│   │   ├── app/              # FastAPI 应用
│   │   │   ├── services/
│   │   │   │   ├── agent/    # LangGraph Agent（核心）
│   │   │   │   ├── llm.py    # LLM 服务
│   │   │   │   └── simulation.py
│   │   │   ├── api/          # REST API
│   │   │   └── models/       # 数据库模型
│   │   └── runners/          # 联邦学习运行时
│   └── frontend/             # React UI
├── libs/fl_core/             # 联邦学习核心库
│   ├── federated/            # Server / Client / 聚合
│   ├── models/               # CNN / ResNet
│   ├── data/                 # 数据加载与划分
│   ├── privacy/              # CKKS 加密
│   ├── compression/          # Top-K 稀疏化
│   └── llm/                  # LLM PEFT 运行时工具
├── configs/                  # 实验配置
├── datasets/llm/             # 本地 LLM SFT / 评测 JSONL 文件
├── models/llm/               # 本地 Hugging Face 兼容 LLM 目录
├── skill/                    # 本地操作说明 skill
├── scripts/                  # Docker 部署脚本
└── .github/workflows/        # CI 流水线
```

## 🤝 贡献 & Roadmap

欢迎提 PR！Figaro 的目标是做一个可读性强、研究友好的联邦学习平台。

**Roadmap**：

**Phase 1：夯实 Agentic 实验平台**
- [x] **Agent 交互体验升级** —— 支持多轮对话微调实验计划，提供实验执行前的Plan Preview。
- [x] **执行与分析透明化** —— 支持实验节点的实时状态追踪，以及 Agent 驱动的运行结果自动化图表解释。
- [x] **配置引擎重构** —— 引入基于 Pydantic/JSON Schema 的严格强校验，彻底修复 `config_schema` 与底层算法实现不一致的问题。
- [x] **高阶实验管理** —— 支持按指标、超参等多维度搜索过滤实验历史，支持配置文件版本控制与 Diff 差异对比。

**Phase 2：基于现有框架的 LLM 联邦 PEFT 微调支持**
- [x] **单机仿真 LoRA/PEFT SFT 路线** —— `task.type=llm_peft_sft` 会分发到专用的单机仿真运行时，加载 Hugging Face 或本地 Causal LM，注入 LoRA Adapter，并基于 JSONL 数据执行每个客户端的监督微调。
- [x] **LLM/PEFT 配置面扩展** —— `config_schema` 与运行时规范化已覆盖基础模型、Tokenizer、最大序列长度、精度、SFT 数据路径/格式、Prompt 模板、LoRA 超参、目标模块、量化模式和 Adapter 续跑路径。
- [x] **JSONL SFT 数据管线** —— 支持 prompt/completion 与 messages 两类 JSONL 格式，支持确定性的客户端数据切分，并提供 plain/chat 风格的文本渲染。
- [x] **Adapter 权重联邦聚合** —— 按客户端样本数聚合 LoRA/PEFT Adapter Tensor，持久化全局 Adapter 产物，记录 SHA-256 血缘，并支持从历史全局 Adapter warm start 续跑。
- [x] **LLM 运行依赖与指标** —— 主项目依赖已包含 `transformers`、`peft`、`accelerate`、`safetensors`、`bitsandbytes`；后端指标已包含 train loss、perplexity、token throughput、adapter size、运行状态、数据摘要和 Adapter artifact lineage。
- [x] **前端与 Agent 的 LLM 运行体验** —— 已在 schema-driven Simulation UI 与 Agent 规划流中开放 LLM 路线，支持按任务类型显示/隐藏字段、LLM 指标曲线、本地模型/数据集选择，以及 Adapter Lineage 产物视图。
- [x] **评测与 Smoke 配置** —— 已增加验证集 JSONL 配置、全局 Adapter 聚合后的逐轮 evaluation loss/perplexity 计算、标准化评测指标，以及 tiny-model LLM PEFT smoke fixture/config。

**Phase 3：平台化与企业级协作**
- [ ] **多租户与细粒度权限** —— 构建多用户隔离的项目空间 (Workspaces)，引入基于角色的访问控制 (RBAC) 和完整的操作审计日志。
- [ ] **生产级调度与容错** —— 实现全局 GPU 资源排队与配额限制，增强分布式环境下的客户端掉线重连与死机容错机制。
- [ ] **云原生基础设施** —— 提供原生的 Kubernetes (K8s) Runner 支持，支持基于排队任务量的 Worker 节点动态扩缩容。
- [ ] **模型资产与血缘管理** —— 建立集中式的产物注册表 (Artifact Registry)，追踪从数据集版本到最终模型权重的完整数据血缘 (Lineage)。
- [ ] **治理与安全合规** —— 提供自动化的隐私合规检查与报告生成（例如审计差分隐私的 $\epsilon$ 参数），确保企业级联邦学习的数据安全。

<p align="center">
  <sub>Figaro 仅用于科研和教学目的。</sub>
</p>
