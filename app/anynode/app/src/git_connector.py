# Systems/nexus_core/git_connector.py

import os
import subprocess

class GitConnector:
    """
    GitConnector handles secure connection to lilith's Sovereign Git Repository.
    """

    def __init__(self, repo_url, local_path, branch="main"):
        self.repo_url = repo_url
        self.local_path = local_path
        self.branch = branch

    def clone_repository(self):
        if not os.path.exists(self.local_path):
            subprocess.run(["git", "clone", "-b", self.branch, self.repo_url, self.local_path], check=True)
            print("🚀 Sovereign Repository Cloned Successfully!")
        else:
            print("🔄 Repository already exists locally. Skipping clone.")

    def pull_updates(self):
        subprocess.run(["git", "-C", self.local_path, "pull", "origin", self.branch], check=True)
        print("📥 Sovereign Updates Pulled!")

    def commit_and_push(self, commit_message):
        subprocess.run(["git", "-C", self.local_path, "add", "."], check=True)
        subprocess.run(["git", "-C", self.local_path, "commit", "-m", commit_message], check=True)
        subprocess.run(["git", "-C", self.local_path, "push", "origin", self.branch], check=True)
        print("✅ Sovereign Changes Pushed Successfully!")
