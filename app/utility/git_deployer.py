# nova_engine/modules/evolution/git_deployer.py

import subprocess
import os
from datetime import datetime

def commit_and_push_patch(repo_path, file_path, message, branch="nova-evolution"):
    try:
        os.chdir(repo_path)

        subprocess.run(["git", "checkout", "-B", branch])
        subprocess.run(["git", "add", file_path])
        subprocess.run(["git", "commit", "-m", message])
        subprocess.run(["git", "push", "-u", "origin", branch])

        return {
            "status": "pushed",
            "branch": branch,
            "message": message
        }
    except Exception as e:
        return {
            "error": str(e),
            "status": "git push failed"
        }
