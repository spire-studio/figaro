# Figaro

**Figaro** — An intelligent federated learning platform with AI-driven experiment orchestration. Describe experiments in natural language, and the agent parses, runs, and compares results automatically.

## Features

### 🤖 Intelligent Experiment Agent

Describe experiments in natural language, the AI agent handles the rest:

```
"Compare CIFAR-10 non-IID with alpha=0.1, 0.3, 0.5"
→ Agent expands to 3 experiments → runs sequentially → outputs comparison table
```

```
"Compare 10 clients vs 20 clients with 5/10/20 rounds"
→ Agent expands to 2×3 = 6 experiments → Cartesian product → matrix view
```

### 🖥️ Simulation

Single-machine FL simulation:
- CIFAR-10 / MNIST datasets
- Non-IID data partitioning (Dirichlet α)
- CNN / ResNet models
- FedAvg aggregation
- CKKS homomorphic encryption
- Top-K sparsification

### 🌐 Distributed

Multi-node distributed training:
- gRPC communication layer
- Server / Worker architecture
- Docker one-click deployment

### 📊 Full-Stack

- **Backend**: FastAPI + PostgreSQL + SQLAlchemy
- **Frontend**: React + Vite + TypeScript
- **Agent UI**: experiment input → real-time progress → comparison results

## ⚙️ Setup

```bash
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
  -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=phoenix postgres:16
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
POSTGRES_HOST=localhost POSTGRES_PORT=5433 POSTGRES_PASSWORD=postgres POSTGRES_DB=phoenix \
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

### Using the Agent

Open the **Agent** tab in the frontend and describe your experiment:
- "Compare CIFAR-10 non-IID with alpha=0.1, 0.3, 0.5"
- "Compare 10 clients vs 20 clients with FedAvg"
- "Test training rounds 10, 20, 50 on accuracy"

The agent will parse your request, run all experiments, and generate a comparison report.

### Distributed deployment

```bash
./scripts/docker-dist-up.sh
```

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

## Project Structure

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
└── scripts/                  # Docker deployment scripts
```
