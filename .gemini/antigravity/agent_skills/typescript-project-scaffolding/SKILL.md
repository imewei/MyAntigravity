---
name: typescript-project-scaffolding
description: Opinionated, production-ready setups for Next.js, Vite, and Node.js.
version: 2.0.0
agents:
  primary: typescript-pro
skills:
- project-setup
- build-configuration
- monorepo-management
- linting-formatting
allowed-tools: [Read, Write, Task, Bash]
---

# TypeScript Project Scaffolding

// turbo-all

# TypeScript Project Scaffolding

Starting correctly to avoid pain later.

---

## Strategy & Templates (Parallel)

// parallel

### Stack Choice

| Type | Stack | Command |
|------|-------|---------|
| **Web App** | Next.js (App Router) | `pnpm create next-app@latest` |
| **SPA** | Vite + React + SWC | `pnpm create vite my-app --template react-ts` |
| **Library** | tsup/Rollup | `pnpm create tsup` |
| **Monorepo** | Turbo + pnpm | `pnpm dlx create-turbo@latest` |

### The "Gold Standard" Config

-   `strict: true` (No `any`).
-   `skipLibCheck: true` (Speed).
-   `moduleResolution: bundler` (Modern).
-   `paths`: `@/*` -> `./src/*` (Clean imports).

// end-parallel

---

## Decision Framework

### Monorepo or Polyrepo?

1.  **Monorepo**: Shared code, atomic commits, consistent tooling. (Use for Products).
2.  **Polyrepo**: Independent lifecycle, strong boundaries. (Use for Microservices/Plugins).

---

## Core Knowledge (Parallel)

// parallel

### Constitutional AI Principles

1.  **Strictness (Target: 100%)**: `noImplicitAny` must be on.
2.  **Speed (Target: <100ms)**: Build tooling (Vite/Esbuild/SWC) over Webpack/Babel.
3.  **Hygiene (Target: 100%)**: Prettier + ESLint on save.

### Quick Reference

-   `pnpm add -D typescript @types/node tsx vitest`
-   `tsc --noEmit` (Type check only).
-   `vitest` (Fast testing).

// end-parallel

---

## Quality Assurance

### Common Bad Habits

| Bad Habit | Fix |
|-----------|-----|
| `any` | Use `unknown` or Generic. |
| Relative Imports `../../../` | Use Paths `@/components`. |
| Committing `dist/` | Add to `.gitignore`. |
| Slow Tests (Jest) | Use Vitest. |

### Setup Checklist

- [ ] `tsconfig.json` (Strict)
- [ ] `.gitignore` (node_modules, dist, .env)
- [ ] ESLint Flat Config
- [ ] Prettier configured
- [ ] Github Actions (CI)
- [ ] `pnpm-lock.yaml` present
