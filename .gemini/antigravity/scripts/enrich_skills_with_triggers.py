import os
import re
from pathlib import Path

SKILLS_DIR = Path(".gemini/antigravity/agent_skills")

# Heuristics for auto-tagging
# (Pattern, [Triggers])
RULES = [
    (r"python", ["file:.py", "project:pyproject.toml", "project:requirements.txt", "keyword:python"]),
    (r"django", ["file:.py", "project:manage.py", "keyword:django", "keyword:web"]),
    (r"fastapi", ["file:.py", "keyword:fastapi", "keyword:api"]),
    (r"flask", ["file:.py", "keyword:flask", "keyword:web"]),
    (r"rust", ["file:.rs", "project:Cargo.toml", "keyword:rust"]),
    (r"julia", ["file:.jl", "project:Project.toml", "keyword:julia"]),
    (r"react", ["file:.jsx", "file:.tsx", "project:package.json", "keyword:react"]),
    (r"typescript", ["file:.ts", "file:.tsx", "project:tsconfig.json", "keyword:typescript"]),
    (r"javascript", ["file:.js", "project:package.json", "keyword:javascript"]),
    (r"node", ["file:.js", "project:package.json", "keyword:node"]),
    (r"docker", ["file:Dockerfile", "file:docker-compose.yml", "keyword:docker"]),
    (r"kubernetes|k8s", ["file:k8s*.yaml", "file:helm/**/*.yaml", "keyword:kubernetes"]),
    (r"terraform", ["file:.tf", "keyword:terraform"]),
    (r"aws|cloud", ["keyword:aws", "keyword:cloud"]),
    (r"ci-cd|pipeline|github-action", ["file:.github/workflows/*.yml", "keyword:ci-cd"]),
    (r"test|qa|quality", ["keyword:testing", "keyword:qa"]),
    (r"security", ["keyword:security", "keyword:audit"]),
    (r"database|sql", ["file:.sql", "keyword:database", "keyword:sql"]),
    (r"visualization|plot", ["keyword:visualization", "keyword:data"]),
    (r"machine-learning|ml|ai", ["keyword:ml", "keyword:ai", "file:.ipynb"]),
]

def parse_frontmatter(content):
    if not content.startswith("---"):
        return None, content
    parts = content.split("---", 2)
    if len(parts) < 3:
        return None, content
    return parts[1], parts[2]

def generate_triggers(name, description):
    triggers = set()
    text = (name + " " + description).lower()
    
    for pattern, rule_triggers in RULES:
        if re.search(pattern, text):
            for t in rule_triggers:
                triggers.add(t)
    
    # Generic fallback
    if not triggers:
        triggers.add(f"keyword:{name}")
        
    return sorted(list(triggers))

def update_skill_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    fm_text, body = parse_frontmatter(content)
    if not fm_text:
        print(f"Skipping {file_path}: No frontmatter")
        return

    # Simple parsing of existing lines
    lines = fm_text.strip().split("\n")
    new_fm_lines = []
    
    name = ""
    description = ""
    has_triggers = False
    
    for line in lines:
        if line.strip().startswith("name:"):
            name = line.split(":", 1)[1].strip()
        if line.strip().startswith("description:"):
            description = line.split(":", 1)[1].strip()
        if line.strip().startswith("triggers:"):
            has_triggers = True
            
    # Generate intelligent triggers
    new_triggers = generate_triggers(name, description)
    
    # Reconstruct frontmatter
    # We filter out existing 'triggers' block if it exists to replace it/merge it
    # But for this batch, we assume we are ADDING optimized triggers
    
    in_triggers_block = False
    for line in lines:
        if line.strip().startswith("triggers:"):
            in_triggers_block = True
            continue 
        if in_triggers_block and line.strip().startswith("-"):
            continue # Skip existing triggers
        if in_triggers_block and not line.strip().startswith("-") and line.strip():
            in_triggers_block = False
        
        if not in_triggers_block:
            new_fm_lines.append(line)

    # Append new triggers
    new_fm_lines.append("triggers:")
    for t in new_triggers:
        new_fm_lines.append(f"- {t}")
        
    new_content = "---\n" + "\n".join(new_fm_lines) + "\n---" + body
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_content)
        
    print(f"Updated {file_path.name} with {len(new_triggers)} triggers")

def main():
    if not SKILLS_DIR.exists():
        print("agent_skills dir not found")
        return

    count = 0
    for skill_file in SKILLS_DIR.glob("**/SKILL.md"):
        update_skill_file(skill_file)
        count += 1
        
    print(f"Processed {count} skills.")

if __name__ == "__main__":
    main()
