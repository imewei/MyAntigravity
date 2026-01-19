import json
import argparse
import re
from pathlib import Path

INDEX_FILE = Path(".gemini/antigravity/skills_index.json")

def load_skills():
    if not INDEX_FILE.exists():
        return {}
    with open(INDEX_FILE, "r") as f:
        data = json.load(f)
    return data.get("skills", {})

def score_skill(skill_data, prompt, files, root_dir):
    triggers = skill_data.get("triggers", [])
    if not triggers:
        return 0
        
    score = 0
    prompt_lower = prompt.lower() if prompt else ""
    
    for trigger in triggers:
        parts = trigger.split(":", 1)
        if len(parts) != 2: continue
        t_type, t_val = parts
        
        if t_type == "keyword":
            if t_val.lower() in prompt_lower:
                score += 10
        elif t_type == "file":
            # Check if ext is in files
            ext = t_val
            if ext.startswith("*"): ext = ext[1:]
            for f in files:
                if f.endswith(ext):
                    score += 15
                    break
        elif t_type == "project":
            # Check if file exists in root
            p_file = root_dir / t_val
            if p_file.exists():
                score += 20
                
    # Boost by description match
    if prompt_lower and prompt_lower in skill_data.get("description", "").lower():
        score += 5
        
    return score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", help="User intent/prompt")
    parser.add_argument("--files", nargs="*", help="List of active files", default=[])
    parser.add_argument("--root", help="Project root", default=".")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    
    root_path = Path(args.root)
    skills = load_skills()
    
    scored = []
    for name, data in skills.items():
        s = score_skill(data, args.prompt, args.files, root_path)
        if s > 0:
            scored.append((s, name, data))
            
    scored.sort(key=lambda x: x[0], reverse=True)
    results = scored[:args.top]
    
    output = []
    for s, name, data in results:
        output.append({
            "name": name,
            "score": s,
            "path": data["path"],
            "description": data["description"]
        })
        
    if args.json:
        print(json.dumps(output, indent=2))
    else:
        print(f"Found {len(output)} relevant skills:")
        for item in output:
             print(f"- {item['name']} (Score: {item['score']})")
             print(f"  Path: {item['path']}")

if __name__ == "__main__":
    main()
