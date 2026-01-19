# Auto-Solve Workflow (v2.0)

// turbo-all

## Phase 1: Capability Discovery (Parallel)

// parallel

1.  **Analyze Request**
    - Agent: `skill-discovery`
    - Action: Analyze the user request "{request}".
    - Command: `uv run .gemini/antigravity/scripts/find_relevant_skills.py --prompt "{request}" --top 3`

2.  **Context Scan**
    - Agent: `skill-discovery`
    - Action: Scan active file types.
    - Command: `git ls-files | awk -F . '{print $NF}' | sort | uniq`

// end-parallel

## Phase 2: Execution Strategy

3.  **Load Specialists**
    - Based on Phase 1 results, identifying the best agent for the job.
    - (Note: In a real run, you would now call the identified agent).

4.  **Triage**
    - If no relevant skill found, default to `python-pro` or `generalist`.
