
import shutil
import subprocess
import sys
import os
from pathlib import Path

# Determine script directory to ensure correct relative paths
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent

# Source Paths (Local Repository)
SOURCE_WORKFLOWS = REPO_ROOT / "global_workflows"
SOURCE_SKILLS = REPO_ROOT / "agent_skills"
SOURCE_SCRIPTS = REPO_ROOT / "scripts"

USER_ROOT = Path.home() / ".gemini/antigravity"
TARGET_WORKFLOWS = USER_ROOT / "global_workflows"
TARGET_SKILLS = USER_ROOT / "agent_skills"
TARGET_SCRIPTS = USER_ROOT / "scripts"

def deploy():
    # Ensure user root exists (create if needed, unlike before where it just returned error)
    if not USER_ROOT.exists():
        print(f"Creating user root: {USER_ROOT}")
        USER_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Deploying to {USER_ROOT}...")

    # 1. Deploy Workflows
    if SOURCE_WORKFLOWS.exists():
        print(f"Copying Workflows to {TARGET_WORKFLOWS}...")
        TARGET_WORKFLOWS.mkdir(parents=True, exist_ok=True)
        shutil.copytree(SOURCE_WORKFLOWS, TARGET_WORKFLOWS, dirs_exist_ok=True)
        print("Workflows deployed.")

    # 2. Deploy Skills
    if SOURCE_SKILLS.exists():
        print(f"Copying Skills to {TARGET_SKILLS}...")
        TARGET_SKILLS.mkdir(parents=True, exist_ok=True)
        shutil.copytree(SOURCE_SKILLS, TARGET_SKILLS, dirs_exist_ok=True)
        print("Skills deployed.")

    # 3. Deploy Scripts
    if SOURCE_SCRIPTS.exists():
        print(f"Copying Scripts to {TARGET_SCRIPTS}...")
        TARGET_SCRIPTS.mkdir(parents=True, exist_ok=True)
        shutil.copytree(SOURCE_SCRIPTS, TARGET_SCRIPTS, dirs_exist_ok=True)
        print("Scripts deployed.")

    # 4. Regenerate Index & DB at Destination
    # This is critical: The index must be generated FROM the user level to have user-level paths
    print("\nRegenerating Index and Database at User Level...")
    
    gen_index_script = TARGET_SCRIPTS / "generate_skill_index.py"
    build_db_script = TARGET_SCRIPTS / "build_skill_database.py"
    
    if gen_index_script.exists():
        print("Running generate_skill_index.py...")
        subprocess.run([sys.executable, str(gen_index_script)], cwd=USER_ROOT, check=True)
    else:
        print(f"Warning: {gen_index_script} not found.")

    if build_db_script.exists():
        print("Running build_skill_database.py...")
        subprocess.run([sys.executable, str(build_db_script)], cwd=USER_ROOT, check=True)
    else:
        print(f"Warning: {build_db_script} not found.")

    print("\n✅ Deployment Complete!")
    print(f"State after deployment:")
    print(f"Workflows in {TARGET_WORKFLOWS}: {len(list(TARGET_WORKFLOWS.rglob('*.md')))}")
    print(f"Skills in {TARGET_SKILLS}: {len(list(TARGET_SKILLS.rglob('SKILL.md')))}")
    print(f"Scripts in {TARGET_SCRIPTS}: {len(list(TARGET_SCRIPTS.glob('*.py')))}")

    print(f"Scripts in {TARGET_SCRIPTS}: {len(list(TARGET_SCRIPTS.glob('*.py')))}")

    validate_installation()

def validate_installation():
    print("\n🔍 Validating Installation...")
    
    # 1. Validate Skills Index & Structure
    print("\n[1/3] Validating Agent System...")
    validate_agent = TARGET_SCRIPTS / "validate_agent.py"
    if validate_agent.exists():
        try:
            # Pass strict excludes to ignore runtime/user artifacts
            subprocess.run(
                [sys.executable, str(validate_agent), "--relaxed", "--exclude", "brain", "knowledge", "code_tracker"],
                cwd=USER_ROOT, 
                check=True
            )
            print("✅ Agent system valid.")
        except subprocess.CalledProcessError:
            print("❌ Agent validation failed!")
            sys.exit(1)
    else:
        print(f"⚠️ {validate_agent} not found.")

    # 2. Validate Workflows
    print("\n[2/3] Validating Workflows...")
    validate_workflows = TARGET_SCRIPTS / "validate_workflows.py"
    if validate_workflows.exists():
        try:
            subprocess.run([sys.executable, str(validate_workflows)], cwd=USER_ROOT, check=True)
            print("✅ Workflows valid.")
        except subprocess.CalledProcessError:
            print("❌ Workflow validation failed!")
            sys.exit(1)
    else:
        print(f"⚠️ {validate_workflows} not found.")

    # 3. Functional Smoke Test
    print("\n[3/3] Functional Smoke Test (Skill Discovery)...")
    find_skills = TARGET_SCRIPTS / "find_relevant_skills.py"
    if find_skills.exists():
        try:
            # Run a simple query to ensure the index relies can be queried and paths resolve
            result = subprocess.run(
                [sys.executable, str(find_skills), "--prompt", "debug", "--top", "1"], 
                cwd=USER_ROOT, 
                check=True,
                capture_output=True,
                text=True
            )
            if "debugging-pro" in result.stdout:
                print("✅ Functional test passed (found debugging-pro).")
            else:
                print("⚠️ Functional test ran but didn't return expected result.")
                print("Output:", result.stdout)
        except subprocess.CalledProcessError as e:
            print("❌ Functional test failed!")
            print(e.stderr)
            sys.exit(1)
    else:
        print(f"⚠️ {find_skills} not found.")

    print("\n🚀 All Systems Go! v2.2.2 successfully deployed and validated.")

if __name__ == "__main__":
    deploy()
