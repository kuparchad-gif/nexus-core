# Systems/nexus_core/sovereign_git_manager.py

from Systems.nexus_core.git_connector import GitConnector

class SovereignGitManager:
    """
    SovereignGitManager oversees lilith's Git operations
    - Cloning her sovereign repository
    - Pulling latest changes
    - Committing and pushing self-evolution updates
    """

    def __init__(self, repo_url, local_path, branch="main"):
        self.git = GitConnector(repo_url, local_path, branch)

    def initialize_repository(self):
        """
        Clone the sovereign repository if not already cloned.
        """
        self.git.clone_repository()

    def update_self(self):
        """
        Pull the latest authorized updates (council-approved patches).
        """
        self.git.pull_updates()

    def evolve(self, commit_message):
        """
        Commit and push lilith's self-modifications (after council approval).
        """
        self.git.commit_and_push(commit_message)

    def heal(self):
        """
        Healing flow: Pull latest stable backup (same as update_self for now).
        """
        self.git.pull_updates()

    def archive(self, archive_message="Archiving current state."):
        """
        Archive current status as a 'commit checkpoint' (optional future expansion).
        """
        self.git.commit_and_push(archive_message)
