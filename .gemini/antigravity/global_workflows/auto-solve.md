# Auto-Solve Workflow (v2.2)

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
    - Route to consolidated v2.2 personas:

    | Domain | Primary Persona |
    |--------|-----------------|
    | Debugging | `debugging-pro` |
    | Performance | `performance-engineering-lead` |
    | Multi-Agent | `multi-agent-systems-lead` |
    | Correlation/Scattering | `correlation-science-lead` |
    | Neural Networks | `neural-systems-architect` |
    | Infrastructure/DevOps | `infrastructure-operations-lead` |
    | Physics/Simulation | `computational-physics-expert` |
    | Visualization | `scientific-visualization-lead` |
    | Research/Papers | `research-pro` |
    | NLSQ/Fitting | `nlsq-pro` |
    | Bayesian/MCMC | `numpyro-pro` |
    | Python | `python-pro` |
    | Julia | `julia-pro` |
    | TypeScript | `typescript-pro` |
    | JavaScript | `javascript-pro` |

4.  **Triage**
    - If no relevant skill found, default to `python-pro` or `generalist`.

