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

**第 2 步 — 配置 `.env`**：
```bash
cp .env.example .env
# 编辑 .env，设置：
#   OPENAI_API_KEY=your-api-key
#   POSTGRES_PASSWORD=postgres
```

**第 3 步 — 后端**：
```bash
POSTGRES_HOST=localhost POSTGRES_PORT=5433 POSTGRES_PASSWORD=postgres POSTGRES_DB=figaro \
  PYTHONPATH=libs:apps/backend/runners \
  uv run uvicorn app.main:app --app-dir apps/backend --host 0.0.0.0 --port 8000 --reload
```

指定 GPU（例如 GPU 1）：
```bash
CUDA_VISIBLE_DEVICES=1 POSTGRES_HOST=localhost ... uv run uvicorn ...
```

**第 4 步 — 前端**（在另一个终端）：
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
│   └── compression/          # Top-K 稀疏化
├── configs/                  # 实验配置
├── scripts/                  # Docker 部署脚本
└── .github/workflows/        # CI 流水线
```

## 🤝 贡献 & Roadmap

欢迎提 PR！Figaro 的目标是做一个可读性强、研究友好的联邦学习平台。

**Roadmap**（暂定，欢迎贡献）：

- [ ] **更丰富的 Agent 规划** —— LangGraph 流水线加入多步反思和失败恢复
- [ ] **更多聚合策略** —— 在现有 FedAvg 之上补充 FedProx、FedAvgM、Scaffold
- [ ] **分布式模式加固** —— 容错、客户端重连、异构 worker 支持
- [ ] **扩展数据集和模型** —— 突破 CIFAR-10 / MNIST 和 CNN / ResNet 的范围
- [ ] **端到端可复现** —— 确定性种子、产物血缘、一键回放
- [ ] **可观测性** —— 单实验指标面板与结构化日志

<p align="center">
  <sub>Figaro 仅用于科研和教学目的。</sub>
</p>
