import sys
from pathlib import Path
import json

# Add scripts dir to path
scripts_dir = Path(__file__).parent
sys.path.append(str(scripts_dir))

import find_relevant_skills

def run_test_cases():
    root_path = Path.cwd()
    files = [] # Simulate no open files for pure prompt testing
    
    # Reload skills each time to be safe
    skills = find_relevant_skills.load_skills()
    
    test_cases = [
        # Prompts that SHOULD match specific skills
        ("repair my code", ["refactor", "debug", "error", "fix"], "Synonym: repair -> fix/refactor"),
        ("rectify the failure", ["fix", "error", "debug"], "Synonym: rectify -> fix"),
        ("runs very slow", ["performance", "speed", "optimize"], "Ambiguity: slow -> performance"),
        ("integartion test", ["test", "qa"], "Typo: integartion -> integration? (Might fail if not in fuzzy limit)"),
        ("dbug this", ["debug", "fix"], "Typo: dbug -> debug"),
        ("make new api", ["create", "backend", "api"], "Complex: make -> create"),
        
        # Cross domain
        ("optimize python build", ["performance", "python", "build", "uv"], "Cross-domain keywords"),
        
        # Negative / Noise
        ("make a sandwich", [], "Negative: Should result in low/no relevant technical skills (or very low score)")
    ]
    
    print(f"{'PROMPT':<30} | {'TOP RESULT':<30} | {'SCORE':<5} | {'STATUS'}")
    print("-" * 80)
    
    passed = 0
    failed = 0
    
    for prompt, expected_keywords, desc in test_cases:
        scored = []
        for name, data in skills.items():
            s, reasons = find_relevant_skills.score_skill(data, prompt, files, root_path)
            if s > 0:
                scored.append((s, name))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        top_skills = scored[:3]
        
        # Validation
        match_found = False
        top_name = "None"
        top_score = 0
        
        if top_skills:
            top_name = top_skills[0][0] # name? No, tuple is (score, name)... wait.
            # In validation script I called it (s, name).
            # Let's check logic above: scored.append((s, name))
            # So x[0] is score. 
            top_score = top_skills[0][0]
            top_name = top_skills[0][1]
            
            # Check if ANY expected keyword is in the top 3 results names or triggers??
            # Simplified: Check if top result name contains expected strings roughly
            # Or better: Check if the *intent* was met.
            
            # Let's check if any of the top 3 skill NAMES contain expected keywords
            for s_val, s_name in top_skills:
                s_name_lower = s_name.lower()
                if any(exp in s_name_lower or exp in str(skills[s_name]) for exp in expected_keywords):
                    match_found = True
                    break
        else:
            if not expected_keywords: # Expected empty
                match_found = True
        
        status = "PASS" if match_found else "FAIL"
        if status == "PASS":
            passed += 1
        else:
            failed += 1
            
        print(f"{prompt:<30} | {top_name:<30} | {top_score:<5} | {status} ({desc})")

    print("-" * 80)
    print(f"Total: {len(test_cases)}. Passed: {passed}. Failed: {failed}.")

if __name__ == "__main__":
    run_test_cases()
