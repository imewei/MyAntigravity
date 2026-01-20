import os
import re
from pathlib import Path
import argparse

def parse_frontmatter(content):
    """
    Manually parses YAML frontmatter without requiring PyYAML.
    """
    if not content.startswith('---\n'):
        return None

    match = re.search(r'^---\n(.*?)\n---', content, re.DOTALL)
    if not match:
        return None

    yaml_block = match.group(1)
    data = {}
    
    current_key = None
    
    for line in yaml_block.split('\n'):
        line = line.rstrip()
        stripped = line.strip()
        
        if not stripped or stripped.startswith('#'):
            continue
            
        if ':' in line and not stripped.startswith('-'):
            parts = line.split(':', 1)
            key = parts[0].strip()
            val = parts[1].strip()
            
            if not val:
                data[key] = []
                current_key = key
            else:
                if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                    val = val[1:-1]
                data[key] = val
                current_key = None
        
        elif stripped.startswith('-'):
            val = stripped[1:].strip()
            if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                    val = val[1:-1]
                    
            if current_key and isinstance(data.get(current_key), list):
                data[current_key].append(val)
                
    return data

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
                frontmatter = parse_frontmatter(content)
                if frontmatter:
                    if 'description' not in frontmatter:
                        issues.append("Missing 'description' in frontmatter")
                else:
                    issues.append("Invalid or missing YAML frontmatter closing")
            except Exception as e:
                issues.append(f"Parse Error: {e}")

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
    parser.add_argument("--workflows-dir", help="Path to workflows directory", default=None)
    args = parser.parse_args()
    
    if args.workflows_dir:
        workflows_dir = Path(args.workflows_dir)
    else:
        # Default to relative path
        workflows_dir = Path(__file__).parent.parent / "global_workflows"
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
