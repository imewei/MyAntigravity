---
name: python-testing-patterns
description: Pytest mastery including fixtures, parameterization, mocking, and property-based testing.
version: 2.0.0
agents:
  primary: test-automator
skills:
- pytest-mastery
- property-based-testing
- async-testing
- fixture-management
allowed-tools: [Read, Write, Task, Bash]
triggers:
- file:.py
- keyword:python
- keyword:qa
- keyword:testing
- project:pyproject.toml
- project:requirements.txt
---

# Python Testing Patterns

// turbo-all

# Python Testing Patterns

Scalable, readable, and powerful testing with Pytest.

---

## Strategy & Techniques (Parallel)

// parallel

### Core Concepts

| Concept | Usage | Example |
|---------|-------|---------|
| **Fixture** | Setup/Teardown | `@pytest.fixture` (DB connection). |
| **Parametrize** | Data-driven | `@pytest.mark.parametrize` (Input tables). |
| **Mock** | Isolate | `unittest.mock.patch` (API calls). |
| **Marker** | Categorize | `@pytest.mark.slow`. |

### Advanced Testing

-   **Hypothesis**: Generate random inputs to find edge cases.
-   **Async**: `pytest-asyncio` for coroutines.
-   **Coverage**: `pytest-cov` for heatmaps.

// end-parallel

---

## Decision Framework

### Fixture Lifecycle (Scope)

1.  **Session**: Once per run (Docker container start).
2.  **Module**: Once per file.
3.  **Function**: Once per test (Default - Database rollback).

---

## Core Knowledge (Parallel)

// parallel

### Constitutional AI Principles

1.  **Isolation (Target: 100%)**: Tests must not affect each other. Use `yield` fixtures.
2.  **Determinism (Target: 100%)**: Seed random numbers.
3.  **Clarity (Target: 100%)**: Test names must explain intent (`test_should_...`).

### Quick Reference

-   `conftest.py`: Shared fixtures.
-   `tmp_path`: Built-in fixture for temp files.
-   `monkeypatch`: Safe environment modifications.
-   `pytest -v -k "login"` (Run specific tests).

// end-parallel

---

## Quality Assurance

### Common Bad Habits

| Bad Habit | Fix |
|-----------|-----|
| Global State | Don't modify module-level vars without reset. |
| Hardcoded Paths | Use `pathlib` and `tmp_path`. |
| Complex Mocks | If mocking is hard, refactor the code (Dependency Injection). |
| Sleeping | Use polling or wait logic, never `time.sleep()`. |

### Pytest Checklist

- [ ] `conftest.py` active
- [ ] Parametrization used for repeated logic
- [ ] Async tests marked properly
- [ ] Codecov configuration active
- [ ] Hypothesis used for math/logic
- [ ] Slow tests marked
