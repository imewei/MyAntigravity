---
description: Scaffold production-ready Rust projects with Cargo tooling and idiomatic patterns
triggers:
- /rust-project
- workflow for rust project
version: 2.0.0
allowed-tools: [Bash, Write, Read, Edit]
agents:
  primary: rust-expert
skills:
- rust-systems-programming
- cargo-mastery
argument-hint: '[project-name] [--mode=quick|standard|enterprise] [--type=bin|lib|workspace]'
---

# Rust Project Scaffolding (v2.0)

// turbo-all

## Phase 1: Cargo Init (Sequential)

1. **Base Project Creation**
   - Action: `cargo new` (bin/lib) or manual workspace setup.
   - Action: Git initialization.

## Phase 2: Component Generation (Parallel)

// parallel

2. **Source Code Structure**
   - Action: Generate `main.rs` / `lib.rs` with idiomatic layout.
   - Action: Create module folders (`api/`, `core/`, `cli/` for Enterpise).

3. **Configuration**
   - Action: Update `Cargo.toml` (dependencies, profiles, workspace members).
   - Action: Create `.gitignore`, `Rustutils.toml` (if needed).

4. **CI/CD & Docs**
   - Action: Generate GitHub Actions workflows.
   - Action: Write `README.md` with build instructions.

// end-parallel

## Phase 3: Quality Assurance (Sequential)

5. **Tooling Check**
   - Action: `cargo check`.
   - Action: `cargo clippy`.
   - Action: `cargo fmt`.
   - Action: `cargo test` (ensure scaffolds pass).
