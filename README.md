<div align="center">
  <h1>🎼 Figaro</h1>
  <p><em>An intelligent federated learning platform with AI-driven experiment orchestration</em></p>
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
    <strong>English</strong> | <a href="./README.zh-CN.md">简体中文</a>
  </p>
</div>

---

🎼 **Figaro** turns natural-language experiment descriptions into reproducible federated learning runs. Describe what you want to compare, and the agent expands, schedules, and summarizes the results for you.

⚡ Built on **FastAPI + LangGraph + PyTorch**, with a React UI for live progress and comparison views.

## 📢 News

- **2026-04-14** 🚦 CI is live — GitHub Actions now runs pytest on the backend and `tsc + vite build` on the frontend for every PR.
- **2026-04-14** 🧹 Major refactor: dropped the legacy attack/defense modules, renamed the project to **Figaro**, tightened reproducibility, and cleaned up log handling.
- **2026-04-10** 🎉 **Figaro v0** — first public cut of the intelligent federated learning platform with the LangGraph-based experiment agent.

## ✨ Key Features

🤖 **Intelligent Experiment Agent** — Describe experiments in natural language. The LangGraph-based agent plans, expands, launches, and summarizes runs automatically.

🖥️ **Simulation Mode** — Single-machine FL with CIFAR-10 / MNIST, Dirichlet non-IID partitioning, CNN / ResNet, FedAvg, CKKS homomorphic encryption, and Top-K sparsification.

🌐 **Distributed Mode** — Multi-node training over gRPC with a Server / Worker architecture and Docker one-click deployment.

📊 **Full-Stack** — FastAPI + PostgreSQL + SQLAlchemy backend, React + Vite + TypeScript frontend, and an Agent tab that shows experiment input, real-time progress, and comparison results.

🔁 **Reproducibility First** — Config-driven runs, pinned seeds, and stored artifacts so every experiment can be replayed.

## 🎯 What the Agent Can Do

```
"Compare CIFAR-10 non-IID with alpha=0.1, 0.3, 0.5"
→ Agent expands to 3 experiments → runs sequentially → outputs comparison table
```

```
"Compare 10 clients vs 20 clients with 5/10/20 rounds"
→ Agent expands to 2×3 = 6 experiments → Cartesian product → matrix view
```

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     React + Vite Frontend                    │
│              (Agent tab · progress · comparison)             │
└───────────────────────────┬──────────────────────────────────┘
                            │ REST / SSE
┌───────────────────────────▼──────────────────────────────────┐
│                      FastAPI Backend                         │
│   ┌────────────┐  ┌──────────────┐  ┌──────────────────┐     │
│   │ LangGraph  │→ │  Experiment  │→ │  Run Service     │     │
│   │   Agent    │  │   Service    │  │ (sim / dist)     │     │
│   └────────────┘  └──────────────┘  └────────┬─────────┘     │
│           ▲                                  │               │
│           │               PostgreSQL         ▼               │
│   ┌───────┴────────┐   ┌────────────┐   ┌──────────────┐     │
│   │  LLM Service   │   │ Job / Run  │   │ fl_core libs │     │
│   │ (OpenAI etc.)  │   │ Repository │   │ FedAvg · CKKS│     │
│   └────────────────┘   └────────────┘   └──────────────┘     │
└──────────────────────────────────────────────────────────────┘
```

## 📚 Table of Contents

- [News](#-news)
- [Key Features](#-key-features)
- [Architecture](#️-architecture)
- [Setup](#️-setup)
- [Quick Start](#-quick-start)
- [Using the Agent](#-using-the-agent)
- [Configuration](#️-configuration)
- [Project Structure](#-project-structure)
- [Contribute & Roadmap](#-contribute--roadmap)

## ⚙️ Setup

```bash
git clone https://github.com/spire-studio/figaro.git
cd figaro
uv sync
```

## 🚀 Quick Start

### Option A: Docker (all-in-one)

```bash
./scripts/docker-dev-up.sh
```

### Option B: Manual startup (recommended for GPU)

**Step 1 — PostgreSQL** (Docker or local):
```bash
docker run -d --name figaro-pg -p 5433:5432 \
  -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=figaro postgres:16
```

**Step 2 — Configure `.env`**:
```bash
cp .env.example .env
# Edit .env and set:
#   OPENAI_API_KEY=your-api-key
#   POSTGRES_PASSWORD=postgres
```

**Step 3 — Backend**:
```bash
POSTGRES_HOST=localhost POSTGRES_PORT=5433 POSTGRES_PASSWORD=postgres POSTGRES_DB=figaro \
  PYTHONPATH=libs:apps/backend/runners \
  uv run uvicorn app.main:app --app-dir apps/backend --host 0.0.0.0 --port 8000 --reload
```

To use a specific GPU (e.g. GPU 1):
```bash
CUDA_VISIBLE_DEVICES=1 POSTGRES_HOST=localhost ... uv run uvicorn ...
```

**Step 4 — Frontend** (in another terminal):
```bash
cd apps/frontend && pnpm install && pnpm dev
```

**Access**:
- Frontend: `http://localhost:5173`
- Backend API: `http://localhost:8000/docs`

### Distributed deployment

```bash
./scripts/docker-dist-up.sh
```

## 🤖 Using the Agent

Open the **Agent** tab in the frontend and describe your experiment:

- "Compare CIFAR-10 non-IID with alpha=0.1, 0.3, 0.5"
- "Compare 10 clients vs 20 clients with FedAvg"
- "Test training rounds 10, 20, 50 on accuracy"

The agent will parse your request, expand it into concrete experiments, run them, and generate a comparison report.

## 🛠️ Configuration

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

## 📁 Project Structure

```
figaro/
├── apps/
│   ├── backend/
│   │   ├── app/              # FastAPI application
│   │   │   ├── services/
│   │   │   │   ├── agent/    # LangGraph Agent (core)
│   │   │   │   ├── llm.py    # LLM service
│   │   │   │   └── simulation.py
│   │   │   ├── api/          # REST API
│   │   │   └── models/       # Database models
│   │   └── runners/          # FL runtime
│   └── frontend/             # React UI
├── libs/fl_core/             # FL core library
│   ├── federated/            # Server / Client / Aggregation
│   ├── models/               # CNN / ResNet
│   ├── data/                 # Data loading & partitioning
│   ├── privacy/              # CKKS encryption
│   └── compression/          # Top-K sparsification
├── configs/                  # Experiment configs
├── scripts/                  # Docker deployment scripts
└── .github/workflows/        # CI pipelines
```

## 🤝 Contribute & Roadmap

PRs welcome! Figaro is meant to be a readable, research-friendly FL platform.

**Roadmap** — tentative, contributions welcome:

- [ ] **Richer agent planning** — multi-step reflection and failure recovery in the LangGraph pipeline
- [ ] **More aggregation strategies** — FedProx, FedAvgM, Scaffold on top of the existing FedAvg baseline
- [ ] **Hardened distributed mode** — fault tolerance, client reconnection, and heterogeneous workers
- [ ] **Expanded datasets & models** — beyond CIFAR-10 / MNIST and CNN / ResNet
- [ ] **End-to-end reproducibility** — deterministic seeds, artifact lineage, and one-click replay
- [ ] **Observability** — per-run metrics dashboard and structured logs

<p align="center">
  <sub>Figaro is for research and educational use.</sub>
</p>
