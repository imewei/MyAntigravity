---
name: debugging-strategies
description: Systematic approaches to solving software defects: Scientific Method, Bisect, Profiling.
version: 2.0.0
agents:
  primary: debugger
skills:
- root-cause-analysis
- hypothesis-testing
- binary-search
- differential-diagnosis
allowed-tools: [Read, Write, Task, Bash]
---

# Debugging Strategies

// turbo-all

# Debugging Strategies

The art and science of solving "it works on my machine".

---

## Strategy & Methods (Parallel)

// parallel

### The Scientific Method

1.  **Observe**: Reproduce the failure. Document input/output.
2.  **Hypothesize**: "I think X caused Y because Z."
3.  **Test**: Create an experiment to prove/disprove.
4.  **Analyze**: Did the test confirm?
5.  **Repeat**: Narrow down scope.

### Core Techniques

-   **Divide & Conquer**: Binary search (comment out half code). Or `git bisect`.
-   **Rubber Duck**: Explain line-by-line to an object/person.
-   **Logging**: Trace execution flow (`console.log` / `print`).
-   **Debugger**: Step-through (`pdb`, `gdb`, VS Code).

// end-parallel

---

## Decision Framework

### Isolation Tactics

1.  **Environment**: Is it Prod only? (Check Config/Data).
2.  **Concurrency**: Race condition? (Check Locks/Async).
3.  **Data**: Specific input? (Minimal Repro Case).
4.  **Time**: Does it happen at midnight? (Cron jobs).

---

## Core Knowledge (Parallel)

// parallel

### Constitutional AI Principles

1.  **Evidence (Target: 100%)**: Don't guess. Prove it.
2.  **Reproduction (Target: 100%)**: Fix confirmed only by failing test -> passing test.
3.  **Root Cause (Target: 100%)**: Don't bandage symptoms. Fix the leak, not the puddle.

### Quick Reference Tools

-   **Python**: `import pdb; pdb.set_trace()`
-   **JS**: `debugger;` statement.
-   **Git**: `git bisect start bad good`.
-   **Network**: Wireshark / Charles / DevTools.

// end-parallel

---

## Quality Assurance

### Common Bad Habits

| Habit | Fix |
|-------|-----|
| "Shotgun Debugging" | Changing things randomly hoping it works. Stop. Think. |
| Ignoring Errors | Read the stack trace. It usually tells you the line. |
| Assuming "Magic" | Code is logical. There is no magic. |
| Console Log Spam | Clean up logs after fixing. |

### Strategy Checklist

- [ ] Reproduction Steps defined
- [ ] Minimal Reproduction Case created
- [ ] Logs/Traces captured
- [ ] Hypothesis written down
- [ ] Change Isolation (One var at a time)
- [ ] Fix includes Regression Test
