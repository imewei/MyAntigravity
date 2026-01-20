---
name: git-advanced-workflows
description: "Advanced Git operations: Interactive Rebase, Bisect, Worktrees, and Recovery."
version: 2.2.1
agents:
  primary: devops-engineer
skills:
- git-mastery
- history-rewrite
- branch-management
- disaster-recovery
allowed-tools: [Read, Write, Task, Bash]
triggers:
- file:.github/workflows/*.yml
- file:.github/CODEOWNERS
- keyword:git
- keyword:github
- keyword:rebase
- keyword:bisect
- keyword:branch
- keyword:worktree
- keyword:merge
- keyword:conflict
---

# Git Advanced Workflows

// turbo-all

# Git Advanced Workflows

Surgical tools for history management and repository hygiene.



## Strategy & Operations (Parallel)

// parallel

### Operation Types

| Operation | Command | Use Case |
|-----------|---------|----------|
| **Cleanup** | `rebase -i` | Squash/Fixup/Reword before push. |
| **Debug** | `bisect` | Find regression commit (Binary Search). |
| **Multitask** | `worktree` | Parallel branches without switching. |
| **Port** | `cherry-pick` | Move specific commit to another branch. |
| **Recover** | `reflog` | Restore lost commits/branches. |

### Rebase Action Table

-   `pick`: Keep.
-   `reword`: Edit message.
-   `edit`: Pause to amend content.
-   `squash`: Meld into previous.
-   `fixup`: Meld (discard log).
-   `drop`: Delete.

// end-parallel

---

## Decision Framework

### Bisect Workflow

1.  `git bisect start`
2.  `git bisect bad` (Current broken state)
3.  `git bisect good <sha>` (Last known working state)
4.  Git checks out middle commit.
5.  Run Test.
6.  `git bisect good` | `git bisect bad`
7.  Repeat until `first bad commit` found.

---

## Core Knowledge (Parallel)

// parallel

### Constitutional AI Principles

1.  **Safety (Target: 100%)**: Never rewrite public history (unless communicated).
2.  **Clarity (Target: 100%)**: Atomic commits with descriptive messages.
3.  **Efficiency (Target: 95%)**: Use `autosquash` for rapid cleanup.

### Quick Reference Commands

-   `git push --force-with-lease` (Safe force).
-   `git worktree add ../feature-b feature-b`.
-   `git reflog` -> `git reset --hard HEAD@{n}`.
-   `git commit --fixup <sha>`.

// end-parallel

---

## Quality Assurance

### Common Pitfalls

| Pitfall | Fix |
|---------|-----|
| Force Pushing Master | Protect main branches. Use `--force-with-lease`. |
| Lost Stash | Use `git rev-list --walk-reflogs stash`. |
| Detached HEAD | Create a branch if you want to save (`git switch -c new-branch`). |
| Merge Conflicts in Rebase | `git rebase --abort` if overwhelmed. |

### Git Checklist

- [ ] Backup branch created before destructive ops
- [ ] `git status` clean before starting
- [ ] Bisect script automated (`git bisect run`)
- [ ] Worktrees pruned after use
- [ ] Reflog checked for "lost" code
