import os
import yaml
import re
from pathlib import Path
import argparse

def validate_workflow(file_path):
    issues = []
    try:
        content = file_path.read_text()
        
        # Check Frontmatter
        if not content.startswith('---\n'):
            issues.append("Missing YAML frontmatter start")
        else:
            try:
                # Extract frontmatter
                match = re.search(r'^---\n(.*?)\n---', content, re.DOTALL)
                if match:
                    frontmatter = yaml.safe_load(match.group(1))
                    if 'description' not in frontmatter:
                        issues.append("Missing 'description' in frontmatter")
                else:
                    issues.append("Invalid or missing YAML frontmatter closing")
            except Exception as e:
                issues.append(f"YAML Error: {e}")

        # Check for turbo directives
        if "// turbo" in content or "// turbo-all" in content:
            pass # Good, modern features used
        
        # Check references to agents (basic heuristic)
        # agents = ["backend-architect", "ai-engineer", "test-automator", ...]
        # if "@" in content: ... could be improved

    except Exception as e:
        issues.append(f"Read Error: {e}")
        
    return issues

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflows-dir", required=True)
    args = parser.parse_args()
    
    workflows_dir = Path(args.workflows_dir)
    if not workflows_dir.exists():
        print(f"Error: {workflows_dir} not found")
        return

    total_files = 0
    total_issues = 0
    
    print(f"🔍 Validating workflows in {workflows_dir}...")
    
    for file_path in workflows_dir.glob("*.md"):
        total_files += 1
        issues = validate_workflow(file_path)
        if issues:
            print(f"\n❌ {file_path.name}:")
            for issue in issues:
                print(f"  - {issue}")
                total_issues += 1
                
    if total_issues == 0:
        print("\n✅ All workflows valid!")
    else:
        print(f"\n⚠️ Found {total_issues} issues in {total_files} files.")

if __name__ == "__main__":
    main()
