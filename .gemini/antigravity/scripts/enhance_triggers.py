#!/usr/bin/env python3
"""Batch enhance skill triggers for better discoverability."""
import os
import re
from pathlib import Path

# Define trigger enhancements
ENHANCEMENTS = {
    # Pro languages - add file extensions
    'golang-pro': ['file:.go', 'keyword:golang', 'keyword:go'],
    'c-pro': ['file:.c', 'file:.h', 'keyword:c'],
    'turing-pro': ['file:.jl', 'keyword:turing', 'keyword:bayesian', 'project:Project.toml'],
    
    # Architecture
    'backend-architect': ['keyword:backend', 'keyword:api', 'keyword:microservice'],
    'graphql-architect': ['keyword:graphql', 'keyword:apollo', 'file:.graphql'],
    'architect-review': ['keyword:architecture', 'keyword:design', 'keyword:review'],
    'microservices-patterns': ['keyword:microservice', 'keyword:saga', 'keyword:cqrs'],
    
    # DevOps
    'observability-engineer': ['keyword:observability', 'keyword:monitoring', 'keyword:traces'],
    'secrets-management': ['keyword:secrets', 'keyword:vault', 'keyword:credentials'],
    'slo-implementation': ['keyword:slo', 'keyword:sla', 'keyword:reliability'],
    'distributed-tracing': ['keyword:tracing', 'keyword:opentelemetry', 'keyword:jaeger'],
    
    # AI/ML
    'rag-implementation': ['keyword:rag', 'keyword:retrieval', 'keyword:vector', 'keyword:embedding'],
    'llm-evaluation': ['keyword:llm', 'keyword:evaluation', 'keyword:benchmark'],
    'prompt-engineer': ['keyword:prompt', 'keyword:llm', 'keyword:gpt'],
    
    # Scientific
    'active-matter': ['keyword:active-matter', 'keyword:swimming', 'file:.jl', 'file:.py'],
    'differential-equations': ['keyword:ode', 'keyword:pde', 'keyword:differential', 'file:.jl'],
    'stochastic-dynamics': ['keyword:stochastic', 'keyword:langevin', 'keyword:brownian'],
    'non-equilibrium-expert': ['keyword:non-equilibrium', 'keyword:transport', 'keyword:irreversible'],
    
    # Julia
    'parallel-computing': ['keyword:parallel', 'keyword:threading', 'keyword:distributed', 'file:.jl'],
    'optimization-patterns': ['keyword:optimization', 'keyword:optim', 'file:.jl', 'project:Project.toml'],
    'variational-inference-patterns': ['keyword:variational', 'keyword:svi', 'keyword:advi', 'file:.jl'],
    
    # Testing/Quality
    'tdd-orchestrator': ['keyword:tdd', 'keyword:test-driven', 'keyword:red-green'],
    'comprehensive-reflection-framework': ['keyword:reflection', 'keyword:review', 'keyword:retrospective'],
    
    # Other
    'legacy-modernizer': ['keyword:legacy', 'keyword:modernize', 'keyword:strangler'],
    'ui-ux-designer': ['keyword:ui', 'keyword:ux', 'keyword:design', 'keyword:figma'],
    'data-analysis': ['keyword:data', 'keyword:analysis', 'keyword:pandas', 'file:.ipynb'],
    'docs-architect': ['keyword:docs', 'keyword:documentation', 'keyword:readme'],
    'git-pr-patterns': ['keyword:pr', 'keyword:pull-request', 'keyword:review'],
    'tutorial-engineer': ['keyword:tutorial', 'keyword:guide', 'keyword:walkthrough'],
    'scientific-computing': ['keyword:scientific', 'keyword:numerical', 'file:.py', 'file:.jl'],
    'command-systems-engineer': ['keyword:cli', 'keyword:command', 'keyword:terminal'],
    'structured-reasoning': ['keyword:reasoning', 'keyword:logic', 'keyword:analysis'],
    'hpc-numerical-coordinator': ['keyword:hpc', 'keyword:cluster', 'keyword:mpi'],
    'turing-model-design': ['keyword:turing', 'keyword:probabilistic', 'file:.jl'],
    'git-advanced-workflows': ['keyword:git', 'keyword:rebase', 'keyword:bisect'],
    
    # Additional
    'gpu-acceleration': ['keyword:gpu', 'keyword:cuda', 'keyword:cupy', 'file:.cu'],
    'api-design-principles': ['keyword:api', 'keyword:rest', 'keyword:openapi'],
    'angular-migration': ['keyword:angular', 'keyword:ngupgrade', 'file:.ts'],
    'context-manager': ['keyword:context', 'keyword:memory', 'keyword:long-context'],
    'auth-implementation-patterns': ['keyword:auth', 'keyword:oauth', 'keyword:jwt'],
    'iterative-error-resolution': ['keyword:error', 'keyword:debug', 'keyword:fix'],
    'monorepo-management': ['keyword:monorepo', 'keyword:turborepo', 'keyword:nx', 'file:turbo.json'],
    'error-handling-patterns': ['keyword:error', 'keyword:retry', 'keyword:circuit-breaker'],
    'md-simulation-setup': ['keyword:lammps', 'keyword:gromacs', 'keyword:molecular'],
    'dx-optimizer': ['keyword:developer-experience', 'keyword:tooling', 'keyword:workflow'],
    'architect-review-framework-migration': ['keyword:migration', 'keyword:framework', 'keyword:upgrade'],
}

def update_skill_triggers(skill_dir: Path, new_triggers: list) -> bool:
    """Update triggers in a SKILL.md file."""
    skill_file = skill_dir / "SKILL.md"
    if not skill_file.exists():
        return False
    
    content = skill_file.read_text()
    
    # Find and update triggers section
    # Pattern: triggers:\n- item\n- item\n (until next key or ---)
    trigger_pattern = r'(triggers:\s*\n)((?:\s*-\s*[^\n]+\n)*)'
    
    match = re.search(trigger_pattern, content)
    if match:
        # Build new triggers block
        new_trigger_block = "triggers:\n"
        for trigger in new_triggers:
            new_trigger_block += f"- {trigger}\n"
        
        # Replace
        new_content = content[:match.start()] + new_trigger_block + content[match.end():]
        skill_file.write_text(new_content)
        return True
    
    return False

def main():
    skills_dir = Path("agent_skills")
    if not skills_dir.exists():
        print("Error: Must run from .gemini/antigravity directory")
        return
    
    updated = 0
    failed = []
    
    for skill_name, triggers in ENHANCEMENTS.items():
        skill_path = skills_dir / skill_name
        if skill_path.exists():
            if update_skill_triggers(skill_path, triggers):
                print(f"✓ Updated {skill_name}")
                updated += 1
            else:
                print(f"✗ Failed to update {skill_name}")
                failed.append(skill_name)
        else:
            print(f"⊘ Skipped {skill_name} (not found)")
    
    print(f"\nUpdated {updated} skills")
    if failed:
        print(f"Failed: {failed}")

if __name__ == "__main__":
    main()
