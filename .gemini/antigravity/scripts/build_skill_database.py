import json
from pathlib import Path
import argparse

SKILLS_INDEX = Path(__file__).parent.parent / "skills_index.json"
OUTPUT_FILE = Path(__file__).parent.parent / "skill_database.json"

def main():
    if not SKILLS_INDEX.exists():
        print(f"Error: {SKILLS_INDEX} not found.")
        return

    with open(SKILLS_INDEX, "r") as f:
        data = json.load(f)
    
    skills = data.get("skills", {})
    
    # Structure: Trigger -> List[SkillName]
    trigger_db = {}
    
    count = 0
    for name, skill_data in skills.items():
        triggers = skill_data.get("triggers", [])
        for t in triggers:
            if t not in trigger_db:
                trigger_db[t] = []
            trigger_db[t].append(name)
            count += 1
            
    # Sort keys
    sorted_db = {k: sorted(v) for k, v in sorted(trigger_db.items())}
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(sorted_db, f, indent=2)
        
    print(f"Built skill database with {len(sorted_db)} unique triggers from {len(skills)} skills.")
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
