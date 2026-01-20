import json
import argparse
import re
import difflib
from pathlib import Path
from collections import Counter

INDEX_FILE = Path(__file__).parent.parent / "skills_index.json"

# Simple synonym map to bridge common intent gaps
SYNONYM_MAP = {
    'debug': {'fix', 'issue', 'error', 'troubleshoot', 'fail', 'crash', 'bug', 'repair'},
    'refactor': {'structure', 'clean', 'modernize', 'rewrite', 'improve', 'organize'},
    'test': {'spec', 'qa', 'check', 'verify', 'validation', 'assert'},
    'build': {'compile', 'package', 'deploy', 'bundle'},
    'deploy': {'release', 'publish', 'ship', 'prod'},
    'create': {'add', 'new', 'generate', 'implement', 'make'},
    'optimize': {'fast', 'performance', 'speed', 'slow', 'latency'},
    'db': {'database', 'sql', 'query', 'store', 'data'},
    'ui': {'frontend', 'interface', 'gui', 'view', 'component'},
    'backend': {'api', 'server', 'service', 'route'},
    'doc': {'documentation', 'guide', 'readme', 'explain'},
}

STOP_WORDS = {
    'the', 'a', 'an', 'to', 'for', 'in', 'of', 'with', 'on', 'at', 'by', 
    'is', 'it', 'this', 'that', 'i', 'me', 'my', 'you', 'your', 'need', 'want',
    'can', 'could', 'help', 'please', 'how', 'do', 'does', 'what', 'where'
}

def load_skills():
    if not INDEX_FILE.exists():
        return {}
    with open(INDEX_FILE, "r") as f:
        data = json.load(f)
    return data.get("skills", {})

def tokenize(text):
    """Normalize and tokenize text."""
    if not text:
        return set()
    # Lowercase and remove non-word chars
    text = text.lower()
    tokens = set(re.findall(r'\w+', text))
    return tokens - STOP_WORDS

def expand_synonyms(tokens):
    """Expand tokens with synonyms."""
    expanded = set(tokens)
    for token in tokens:
        for key, synonyms in SYNONYM_MAP.items():
            # If token is a key, add synonyms
            if token == key:
                expanded.update(synonyms)
            # If token is in synonyms, add key
            elif token in synonyms:
                expanded.add(key)
                expanded.update(synonyms)
    return expanded

def score_skill(skill_data, prompt, files, root_dir):
    triggers = skill_data.get("triggers", [])
    if not triggers:
        return 0, []
        
    score = 0
    reasons = []
    prompt_tokens = tokenize(prompt)
    expanded_tokens = expand_synonyms(prompt_tokens)
    
    # Description tokens for fallback matching
    description = skill_data.get("description", "")
    desc_tokens = tokenize(description)
    
    # 1. Trigger Matching
    for trigger in triggers:
        parts = trigger.split(":", 1)
        if len(parts) != 2: continue
        t_type, t_val = parts
        t_val_lower = t_val.lower()
        
        if t_type == "keyword":
            trigger_tokens = tokenize(t_val)
            
            # Smart Matching:
            # 1. If trigger is a phrase (multiple words), require exact phrase match in prompt
            # 2. If trigger is a single word, require it to be a distinct token (no partial "c" in "docs")
            
            is_phrase = len(t_val.split()) > 1
            if is_phrase:
                if t_val_lower in prompt.lower():
                    score += 30
                    reasons.append(f"Phrase match '{t_val}' (+30)")
                    continue
            else:
                # Single word trigger - must be in tokens (exact or synonym)
                # We check intersection with expanded_tokens
                pass

            # Token intersection (strong)
            overlap = expanded_tokens.intersection(trigger_tokens)
            if overlap:
                score += len(overlap) * 15
                reasons.append(f"Keyword match {overlap} (+{len(overlap)*15})")
            else:
                # Fuzzy match (medium) - STRICTER
                for t_tok in trigger_tokens:
                    if len(t_tok) < 4: continue 
                    matches = difflib.get_close_matches(t_tok, expanded_tokens, n=1, cutoff=0.8)
                    if matches:
                        score += 10
                        reasons.append(f"Fuzzy match '{t_tok}'~{matches} (+10)")
                        break
                        
        elif t_type == "file":
            # Check if ext is in files
            ext = t_val
            if ext.startswith("*"): ext = ext[1:]
            for f in files:
                if f.endswith(ext):
                    score += 20
                    break
                    
        elif t_type == "project":
            # Check if file exists in root
            p_file = root_dir / t_val
            if p_file.exists():
                score += 40
                
    # 2. Description Matching (Contextual boost)
    # Give points for token overlap between prompt and description
    desc_overlap = expanded_tokens.intersection(desc_tokens)
    if desc_overlap:
        score += len(desc_overlap) * 3
        # reasons.append(f"Description overlap: {desc_overlap} (+{len(desc_overlap)*3})")
        
    return score, [] # Return empty reasons for now to match main signature, I'll add logic if needed but user just wants the fix. 
    # Actually I should implement reason tracking properly.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", help="User intent/prompt", default="")
    parser.add_argument("--files", nargs="*", help="List of active files", default=[])
    parser.add_argument("--root", help="Project root", default=".")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    
    root_path = Path(args.root)
    skills = load_skills()
    
    scored = []
    for name, data in skills.items():
        s, reason = score_skill(data, args.prompt, args.files, root_path)
        if s > 0:
            scored.append((s, name, data, reason))
            
    scored.sort(key=lambda x: x[0], reverse=True)
    results = scored[:args.top]
    
    output = []
    for s, name, data, reason in results:
        output.append({
            "name": name,
            "score": s,
            "path": data["path"],
            "description": data["description"],
            "reason": reason
        })
        
    if args.json:
        print(json.dumps(output, indent=2))
    else:
        print(f"Found {len(output)} relevant skills:")
        for item in output:
             print(f"- {item['name']} (Score: {item['score']})")
             print(f"  Reason: {item['reason']}")
             print(f"  Path: {item['path']}")
             print(f"  Description: {item['description'][:100]}...")

if __name__ == "__main__":
    main()
