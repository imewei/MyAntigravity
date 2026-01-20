# MyAntigravity

> **Intelligent AI Agent Skills & Workflow Ecosystem for Scientific Computing**

A modular, extensible platform providing 127 specialized AI agent skills and 17 automated workflows for scientific computing, software engineering, and research applications.

## 🎯 What is MyAntigravity?

MyAntigravity is a skill-based AI augmentation system designed for:

- **Scientific Computing**: JAX/NumPy optimization, Bayesian inference, molecular dynamics
- **Software Engineering**: Multi-language expertise (Python, Julia, TypeScript, Rust, Go)
- **DevOps & Infrastructure**: CI/CD, Kubernetes, observability, security
- **Research**: Systematic reviews, paper implementation, quality assessment

---

## 📊 Quick Stats (v2.2.2)

| Component | Count | Description |
|-----------|-------|-------------|
| **Skills** | 127 | Specialized AI personas & capabilities |
| **Workflows** | 17 | Automated multi-step processes |
| **Scripts** | 10 | Utility tools for management |
| **Trigger Quality** | 98% | Multi-trigger discoverability |

---

## 🏗️ Architecture

```
.gemini/antigravity/
├── agent_skills/           # 127 specialized AI skills
│   ├── python-pro/         # Python expertise + testing + async
│   ├── julia-pro/          # Julia HPC + performance tuning
│   ├── jax-optimization-pro/ # JAX High-Performance Computing
│   ├── jax-bayesian-pro/   # NumPyro/BlackJAX Inference
│   ├── jax-diffeq-pro/     # Diffrax/Neural ODEs
│   └── ...
├── global_workflows/       # 17 automated workflows
│   ├── auto-solve.md       # Meta-orchestrator for routing
│   ├── commit.md           # Smart git commit workflow
│   └── ...
├── scripts/                # Management utilities
│   ├── generate_skill_index.py
│   ├── find_relevant_skills.py # Key: Fuzzy/Token Matching
│   ├── build_skill_database.py # Key: Trigger Aggregation
│   └── enhance_triggers.py
├── skills_index.json       # Searchable skill registry
└── skill_database.json     # Flattened Trigger DB for verification
```

---

## 🚀 Installation

### Automated Deployment

Run the included deployment script to install the ecosystem to your user level (`~/.gemini/antigravity`). This script automatically:
1.  **Deploys** all skills, workflows, and scripts.
2.  **Regenerates** the skill index with correct user-level paths.
3.  **Validates** the installation (Agent integrity, Workflow structure, Functional smoke test).

```bash
uv run python3 .gemini/antigravity/scripts/deploy_to_user_level.py
```

If successful, you will see `🚀 All Systems Go!`.

---

## 🛠️ Using Skills

### Automatic Discovery

Skills are automatically discovered based on:
- **File extensions**: `.py`, `.jl`, `.ts`, `.go`, etc.
- **Keywords**: "bayesian", "optimize", "debug", etc.
- **Project files**: `pyproject.toml`, `Project.toml`, `Cargo.toml`

### Manual Invocation

Mention a skill by name to invoke it directly:

```
Use python-pro to write async data processing
Use nlsq-pro to fit this scattering model
Use research-pro to review this methodology
```

### Key Unified Personas (v2.2.2)

| Persona | Domain | Merged Skills |
|---------|--------|---------------|
| `debugging-pro` | Debugging | 4 debugging skills |
| `performance-engineering-lead` | Optimization | 3 performance skills |
| `neural-systems-architect` | Deep Learning | 5 neural skills |
| `correlation-science-lead` | Scattering | 5 correlation skills |
| `infrastructure-operations-lead` | DevOps | 5 infrastructure skills |
| `research-pro` | Research | 3 research skills |

---

## 📋 Using Workflows

### Slash Command Invocation

```
/commit --split          # Smart git commit with atomic splits
/code-explain            # Detailed code explanation
/full-review             # Multi-agent code review
/speckit-specify         # Feature specification
/double-check            # Multi-dimensional validation
```

### Auto-Solve Routing

The `auto-solve` workflow automatically routes requests to appropriate skills:

```
"Debug this memory leak" → debugging-pro
"Optimize GPU utilization" → gpu-acceleration
"Fit SAXS model" → nlsq-pro
"Bayesian parameter estimation" → numpyro-pro
```

---

## 📜 Available Scripts

| Script | Purpose |
|--------|---------|
| `generate_skill_index.py` | Regenerate `skills_index.json` |
| `find_relevant_skills.py` | Search for skills by query (Enhanced v2.3 with fuzzy/token matching) |
| `build_skill_database.py` | Create centralized `skill_database.json` for analysis |
| `test_skill_discovery_edges.py` | Verify skill discovery against edge cases |
| `enhance_triggers.py` | Batch-update skill triggers |
| `validate_agent.py` | Validate agent/skill syntax |
| `validate_workflows.py` | Validate workflow structure |
| `validate_plugin_syntax.py` | Check cross-references |
| `deploy_to_user_level.py` | Copy to `~/.gemini` |

---

## 🔧 Customization

### Adding a New Skill

1. Create directory: `.gemini/antigravity/agent_skills/my-skill/`
2. Create `SKILL.md` with YAML frontmatter:

```yaml
---
name: my-skill
description: What this skill does
version: 2.2.2
triggers:
- keyword:my-keyword
- file:.ext
- project:config.file
---

# My Skill

Instructions for the AI agent...
```

3. Regenerate index:
```bash
python3 scripts/generate_skill_index.py
```

### Trigger Types

| Type | Example | Activation |
|------|---------|------------|
| `file:` | `file:.py` | File extension match |
| `keyword:` | `keyword:optimize` | Keyword in query |
| `project:` | `project:pyproject.toml` | Project file presence |

---

## 📈 Ecosystem Health

### Trigger Quality Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Multi-trigger (≥3) | >80% | **95%** ✅ |
| File/Project triggers | >50% | **61%** ✅ |
| Keyword-only | <10% | **0%** ✅ |

### Validation Commands

```bash
# Regenerate and validate index
cd ~/.gemini/antigravity
python3 scripts/generate_skill_index.py

# Find relevant skills for a query
python3 scripts/find_relevant_skills.py --prompt "Bayesian optimization" --top 5
```

---

## 📚 Documentation

- **Knowledge Items**: See `~/.gemini/antigravity/knowledge/` for curated domain knowledge
- **Skill Details**: Each skill contains inline documentation in `SKILL.md`
- **Workflow Guide**: Workflows are self-documenting with step-by-step instructions

---

## 📄 License

Private repository. All rights reserved.