
import shutil
from pathlib import Path

# Determine script directory to ensure correct relative paths
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent

# Source Paths (Local Repository)
SOURCE_WORKFLOWS = REPO_ROOT / "global_workflows"
SOURCE_SKILLS = REPO_ROOT / "agent_skills"
SOURCE_SCRIPTS = REPO_ROOT / "scripts"
SOURCE_INDEX = REPO_ROOT / "skills_index.json"

USER_ROOT = Path.home() / ".gemini/antigravity"
TARGET_WORKFLOWS = USER_ROOT / "global_workflows"
TARGET_SKILLS = USER_ROOT / "agent_skills"
TARGET_SCRIPTS = USER_ROOT / "scripts"
TARGET_INDEX = USER_ROOT / "skills_index.json"

def deploy():
    if not USER_ROOT.exists():
        print(f"Error: User root {USER_ROOT} does not exist.")
        return

    print(f"Deploying to {USER_ROOT}...")

    # 1. Deploy Workflows
    if SOURCE_WORKFLOWS.exists():
        print(f"Copying Workflows to {TARGET_WORKFLOWS}...")
        TARGET_WORKFLOWS.mkdir(exist_ok=True)
        # Copy contents using copytree with dirs_exist_ok=True
        shutil.copytree(SOURCE_WORKFLOWS, TARGET_WORKFLOWS, dirs_exist_ok=True)
        print("Workflows deployed.")
    else:
        print("No source workflows found.")

    # 2. Deploy Skills
    if SOURCE_SKILLS.exists():
        print(f"Copying Skills to {TARGET_SKILLS}...")
        TARGET_SKILLS.mkdir(exist_ok=True)
        shutil.copytree(SOURCE_SKILLS, TARGET_SKILLS, dirs_exist_ok=True)
        print("Skills deployed.")
    else:
        print("No source skills found.")

    # 3. Deploy Scripts
    if SOURCE_SCRIPTS.exists():
        print(f"Copying Scripts to {TARGET_SCRIPTS}...")
        TARGET_SCRIPTS.mkdir(exist_ok=True)
        shutil.copytree(SOURCE_SCRIPTS, TARGET_SCRIPTS, dirs_exist_ok=True)
        print("Scripts deployed.")
    else:
        print("No source scripts found.")

    # 4. Deploy Skills Index
    if SOURCE_INDEX.exists():
        print(f"Copying Skills Index to {TARGET_INDEX}...")
        shutil.copy2(SOURCE_INDEX, TARGET_INDEX)
        print("Skills Index deployed.")
    else:
        print("No source skills_index.json found.")

    print("\nState after deployment:")
    print(f"Workflows in {TARGET_WORKFLOWS}: {len(list(TARGET_WORKFLOWS.rglob('*.md')))}")
    print(f"Skills in {TARGET_SKILLS}: {len(list(TARGET_SKILLS.rglob('SKILL.md')))}")
    print(f"Scripts in {TARGET_SCRIPTS}: {len(list(TARGET_SCRIPTS.glob('*.py')))}")

if __name__ == "__main__":
    deploy()
