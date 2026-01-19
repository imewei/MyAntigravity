---
description: Scaffold production-ready Python projects with modern tooling (uv, ruff, pytest)
triggers:
- /python-scaffold
- scaffold production ready python projects
allowed-tools: [Read, Task, Bash]
version: 2.0.0
agents:
  primary: python-developer
skills:
- python-pro
- project-scaffolding-best-practices
argument-hint: '[project-name] [--mode=quick|standard|enterprise] [--type=fastapi|django|lib|cli]'
---

# Python Project Scaffolding (v2.0)

// turbo-all

## Phase 1: Initialization (Sequential)

1. **UV Init**
   - Action: `uv init`, `git init`.
   - Action: Create `.gitignore` (Python/MacOS/IDE standards).
   - Action: Create virtual env (`uv venv`).

## Phase 2: Structural Generation (Parallel)

// parallel

2. **Core Directory Structure**
   - Action: Create `src/`, `tests/`, `docs/`.
   - Action: Create specific folders based on `--type` (e.g., `app/routers` for FastAPI).

3. **Tool Configuration**
   - Action: Generate `pyproject.toml` (Ruff, Pytest settings).
   - Action: Generate `.env.example`, `Makefile`, `Dockerfile`.

4. **Documentation**
   - Action: Generate `README.md` with setup/usage instructions.
   - Action: Generate `CONTRIBUTING.md`.

// end-parallel

## Phase 3: Dependencies (Sequential)

5. **Dependency Installation**
   - Action: `uv add` runtime deps (fastapi, typer, etc.).
   - Action: `uv add --dev` dev deps (ruff, pytest, mypy).

## Phase 4: Verification (Parallel)

// parallel

6. **Sanity Check**
   - Action: Run `pytest`.
   - Action: Run `ruff check`.

7. **Build Check**
   - Action: Verify Docker build (if Standard/Enterprise).

// end-parallel
