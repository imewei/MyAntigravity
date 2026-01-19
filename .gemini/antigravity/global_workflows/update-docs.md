---
description: Update documentation from code analysis
triggers:
- /update-docs
- update documentation
version: 2.0.0
allowed-tools: [Read, Task, Bash]
agents:
  primary: technical-writer
skills:
- documentation-automation
argument-hint: '[--dry-run]'
---

# Docs Updater (v2.0)

// turbo-all

## Phase 1: Discovery (Parallel)

// parallel

1.  **Change Detection**
    - Action: `git diff` since last tag.

2.  **API Scan**
    - Action: Extract current API signatures.

// end-parallel

## Phase 2: Update (Parallel)

// parallel

3.  **Sync References**
    - Action: Update Sphinx/MkDocs autodoc.

4.  **Sync README**
    - Action: Update Install/Config sections.

5.  **Sync Examples**
    - Action: Verify and update code snippets.

// end-parallel

## Phase 3: Build

6.  **Verify**
    - Action: Build docs (html). Check warnings.
