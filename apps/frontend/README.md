# Figaro Frontend (React + OpenAPI)

## Setup

```bash
cd apps/frontend
pnpm install
```

## Generate OpenAPI Types

Start backend first (`uv run uvicorn app.main:app --reload --app-dir apps/backend`), then:

```bash
pnpm run gen:api
```

This updates `src/api/openapi.d.ts` from FastAPI OpenAPI schema.

## Run Dev Server

```bash
pnpm run dev
```

Frontend uses `VITE_API_BASE_URL` (default: `http://localhost:8000`).

## Pages/Features

- Jobs list/create/update/copy
- Save current task config
- Run start/stop
- Run status, logs, artifacts polling
