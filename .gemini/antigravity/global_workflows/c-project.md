---
description: Scaffold production-ready C projects with memory safety tools
triggers:
- /c-project
- workflow for c project
version: 2.0.0
allowed-tools: [Bash, Write, Read, Edit]
agents:
  primary: c-developer
skills:
- systems-programming
- c-patterns
argument-hint: '[project-name] [--mode=quick|standard|enterprise]'
---

# C Project Scaffolder (v2.0)

// turbo-all

## Phase 1: Structure (Parallel)

// parallel

1.  **Directories**
    - Action: Create `src/`, `include/`, `tests/`.

2.  **Build System**
    - Action: Generate `Makefile` (with Valgrind targets) and `CMakeLists.txt`.

3.  **Source Templates**
    - Action: Generate `main.c`, `logger.c`, `project.h`.

// end-parallel

## Phase 2: Configuration (Sequential)

4.  **Tooling Setup**
    - Action: Generate `.gitignore` and `.clang-format`.

5.  **Documentation**
    - Action: Write `README.md` with build instructions.

## Phase 3: Verification (Sequential)

6.  **Build Check**
    - Action: Run `make`.

7.  **Memory Safety Check**
    - Action: Run `valgrind ./bin/main` (if available).
