---
description: Generate complete TypeScript project structures with modern tooling
triggers:
- /typescript-scaffold
- workflow for typescript scaffold
allowed-tools: [Read, Task, Bash]
version: 2.0.0
agents:
  primary: frontend-developer
skills:
- typescript-pro
- modern-web-tooling
argument-hint: '[project-name] [--mode=quick|standard|comprehensive] [--type=next|vite|node|lib]'
---

# TypeScript Scaffolding (v2.0)

// turbo-all

## Phase 1: Initialization (Sequential)

1. **Base Creation**
   - Action: `pnpm create` (vite/next/etc) or `pnpm init`.
   - Action: `git init`.

## Phase 2: Configuration & Tooling (Parallel)

// parallel

2. **TypeScript Config**
   - Action: Generate `tsconfig.json` (Strict checks, paths).

3. **Linter/Formatter**
   - Action: Configure `eslint.config.js`, `prettierrc`.
   - Action: Setup `.vscode/settings.json`.

4. **Testing Setup**
   - Action: Configure `vitest.config.ts`, `setupTests.ts`.

5. **Project Structure**
   - Action: Create strict folder hierarchy (`src/features`, `src/lib`, etc.).

// end-parallel

## Phase 3: Finalization (Sequential)

6. **Install & Verify**
   - Action: `pnpm install`.
   - Action: `pnpm run type-check`.
   - Action: `pnpm test`.
