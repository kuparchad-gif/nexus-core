#!/usr/bin/env python3
"""
GitHub Client for Viren Platinum Edition
Manages GitHub repositories, commits, and authentication
"""

import os
import json
import logging
import base64
import requests
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
logger = logging.getLogger("GitHubClient")

class GitHubClient:
    """
    GitHub client for managing repositories and authentication
    Supports multiple accounts (up to 10)
    """
    
    def __init__(self, config_path: str = None):
        """Initialize the GitHub client"""
        self.config_path = config_path or os.path.join("config", "github_config.json")
        self.accounts = {}
        self.active_account = None
        self.max_accounts = 10
        
        # Load configuration
        self._load_config()
    
    def _load_config(self):
        """Load GitHub configuration"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    # Store encrypted credentials
                    self.accounts = config.get("accounts", {})
                    self.active_account = config.get("active_account")
                    logger.info(f"Loaded GitHub configuration with {len(self.accounts)} accounts")
            except Exception as e:
                logger.error(f"Error loading GitHub configuration: {e}")
                self.accounts = {}
                self.active_account = None
        else:
            logger.info("GitHub configuration not found")
            self.accounts = {}
            self.active_account = None
    
    def _save_config(self):
        """Save GitHub configuration"""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        try:
            config = {
                "accounts": self.accounts,
                "active_account": self.active_account
            }
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info("GitHub configuration saved")
        except Exception as e:
            logger.error(f"Error saving GitHub configuration: {e}")
    
    def _encrypt_password(self, password: str) -> str:
        """
        Simple encryption for password storage
        Note: This is not secure for production use
        """
        return base64.b64encode(password.encode()).decode()
    
    def _decrypt_password(self, encrypted: str) -> str:
        """
        Simple decryption for password retrieval
        Note: This is not secure for production use
        """
        return base64.b64decode(encrypted.encode()).decode()
    
    def add_account(self, username: str, password: str, token: str = None) -> bool:
        """
        Add a GitHub account
        
        Args:
            username: GitHub username
            password: GitHub password
            token: GitHub personal access token (optional)
            
        Returns:
            True if successful, False otherwise
        """
        if len(self.accounts) >= self.max_accounts:
            logger.warning(f"Maximum number of accounts ({self.max_accounts}) reached")
            return False
        
        # Check if account already exists
        if username in self.accounts:
            logger.warning(f"Account {username} already exists")
            return False
        
        # Encrypt password
        encrypted_password = self._encrypt_password(password)
        
        # Add account
        self.accounts[username] = {
            "username": username,
            "password": encrypted_password,
            "token": token,
            "repos": []
        }
        
        # Set as active account if first account
        if not self.active_account:
            self.active_account = username
        
        # Save configuration
        self._save_config()
        logger.info(f"Added GitHub account: {username}")
        
        # Test connection
        if self.test_connection(username):
            # Fetch repositories
            self.fetch_repositories(username)
            return True
        else:
            logger.warning(f"Failed to connect to GitHub with account {username}")
            return False
    
    def remove_account(self, username: str) -> bool:
        """
        Remove a GitHub account
        
        Args:
            username: GitHub username
            
        Returns:
            True if successful, False otherwise
        """
        if username not in self.accounts:
            logger.warning(f"Account {username} not found")
            return False
        
        # Remove account
        del self.accounts[username]
        
        # Update active account if needed
        if self.active_account == username:
            self.active_account = next(iter(self.accounts.keys())) if self.accounts else None
        
        # Save configuration
        self._save_config()
        logger.info(f"Removed GitHub account: {username}")
        
        return True
    
    def set_active_account(self, username: str) -> bool:
        """
        Set the active GitHub account
        
        Args:
            username: GitHub username
            
        Returns:
            True if successful, False otherwise
        """
        if username not in self.accounts:
            logger.warning(f"Account {username} not found")
            return False
        
        self.active_account = username
        self._save_config()
        logger.info(f"Set active GitHub account: {username}")
        
        return True
    
    def get_accounts(self) -> List[str]:
        """
        Get list of GitHub accounts
        
        Returns:
            List of account usernames
        """
        return list(self.accounts.keys())
    
    def get_active_account(self) -> Optional[str]:
        """
        Get active GitHub account
        
        Returns:
            Active account username or None
        """
        return self.active_account
    
    def test_connection(self, username: str = None) -> bool:
        """
        Test connection to GitHub
        
        Args:
            username: GitHub username or None for active account
            
        Returns:
            True if successful, False otherwise
        """
        username = username or self.active_account
        if not username or username not in self.accounts:
            logger.warning("No valid account specified")
            return False
        
        account = self.accounts[username]
        
        try:
            # Try with token first if available
            if account.get("token"):
                headers = {"Authorization": f"token {account['token']}"}
                response = requests.get("https://api.github.com/user", headers=headers)
                if response.status_code == 200:
                    logger.info(f"Successfully connected to GitHub with token for {username}")
                    return True
            
            # Try with username/password
            password = self._decrypt_password(account["password"])
            response = requests.get(
                "https://api.github.com/user",
                auth=(username, password)
            )
            
            if response.status_code == 200:
                logger.info(f"Successfully connected to GitHub with password for {username}")
                return True
            else:
                logger.warning(f"Failed to connect to GitHub: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error connecting to GitHub: {e}")
            return False
    
    def fetch_repositories(self, username: str = None) -> List[Dict[str, Any]]:
        """
        Fetch repositories for a GitHub account
        
        Args:
            username: GitHub username or None for active account
            
        Returns:
            List of repositories
        """
        username = username or self.active_account
        if not username or username not in self.accounts:
            logger.warning("No valid account specified")
            return []
        
        account = self.accounts[username]
        
        try:
            # Try with token first if available
            if account.get("token"):
                headers = {"Authorization": f"token {account['token']}"}
                response = requests.get("https://api.github.com/user/repos", headers=headers)
                if response.status_code == 200:
                    repos = response.json()
                    account["repos"] = repos
                    self._save_config()
                    logger.info(f"Fetched {len(repos)} repositories for {username}")
                    return repos
            
            # Try with username/password
            password = self._decrypt_password(account["password"])
            response = requests.get(
                "https://api.github.com/user/repos",
                auth=(username, password)
            )
            
            if response.status_code == 200:
                repos = response.json()
                account["repos"] = repos
                self._save_config()
                logger.info(f"Fetched {len(repos)} repositories for {username}")
                return repos
            else:
                logger.warning(f"Failed to fetch repositories: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error fetching repositories: {e}")
            return []
    
    def get_repositories(self, username: str = None) -> List[Dict[str, Any]]:
        """
        Get repositories for a GitHub account
        
        Args:
            username: GitHub username or None for active account
            
        Returns:
            List of repositories
        """
        username = username or self.active_account
        if not username or username not in self.accounts:
            logger.warning("No valid account specified")
            return []
        
        account = self.accounts[username]
        
        # Return cached repositories or fetch if empty
        if not account.get("repos"):
            return self.fetch_repositories(username)
        
        return account.get("repos", [])
    
    def clone_repository(self, repo_name: str, local_path: str, username: str = None) -> bool:
        """
        Clone a GitHub repository
        
        Args:
            repo_name: Repository name (owner/repo)
            local_path: Local path to clone to
            username: GitHub username or None for active account
            
        Returns:
            True if successful, False otherwise
        """
        username = username or self.active_account
        if not username or username not in self.accounts:
            logger.warning("No valid account specified")
            return False
        
        account = self.accounts[username]
        password = self._decrypt_password(account["password"])
        
        try:
            # Construct clone URL with credentials
            if "/" not in repo_name:
                repo_name = f"{username}/{repo_name}"
            
            clone_url = f"https://{username}:{password}@github.com/{repo_name}.git"
            
            # Clone repository
            import subprocess
            result = subprocess.run(
                ["git", "clone", clone_url, local_path],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully cloned repository {repo_name}")
                return True
            else:
                logger.warning(f"Failed to clone repository: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Error cloning repository: {e}")
            return False
    
    def create_repository(self, name: str, description: str = "", private: bool = False, 
                         username: str = None) -> Optional[Dict[str, Any]]:
        """
        Create a new GitHub repository
        
        Args:
            name: Repository name
            description: Repository description
            private: Whether the repository is private
            username: GitHub username or None for active account
            
        Returns:
            Repository data if successful, None otherwise
        """
        username = username or self.active_account
        if not username or username not in self.accounts:
            logger.warning("No valid account specified")
            return None
        
        account = self.accounts[username]
        
        try:
            # Prepare repository data
            repo_data = {
                "name": name,
                "description": description,
                "private": private
            }
            
            # Try with token first if available
            if account.get("token"):
                headers = {
                    "Authorization": f"token {account['token']}",
                    "Accept": "application/vnd.github.v3+json"
                }
                response = requests.post(
                    "https://api.github.com/user/repos",
                    headers=headers,
                    json=repo_data
                )
                if response.status_code == 201:
                    repo = response.json()
                    logger.info(f"Created repository {name}")
                    return repo
            
            # Try with username/password
            password = self._decrypt_password(account["password"])
            response = requests.post(
                "https://api.github.com/user/repos",
                auth=(username, password),
                json=repo_data
            )
            
            if response.status_code == 201:
                repo = response.json()
                logger.info(f"Created repository {name}")
                return repo
            else:
                logger.warning(f"Failed to create repository: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error creating repository: {e}")
            return None
    
    def push_to_repository(self, local_path: str, message: str, username: str = None) -> bool:
        """
        Commit and push changes to a GitHub repository
        
        Args:
            local_path: Local repository path
            message: Commit message
            username: GitHub username or None for active account
            
        Returns:
            True if successful, False otherwise
        """
        username = username or self.active_account
        if not username or username not in self.accounts:
            logger.warning("No valid account specified")
            return False
        
        try:
            import subprocess
            
            # Change to repository directory
            cwd = os.getcwd()
            os.chdir(local_path)
            
            # Add all changes
            subprocess.run(["git", "add", "."], check=True)
            
            # Commit changes
            subprocess.run(["git", "commit", "-m", message], check=True)
            
            # Push changes
            subprocess.run(["git", "push"], check=True)
            
            # Change back to original directory
            os.chdir(cwd)
            
            logger.info(f"Successfully pushed changes to repository")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Git command failed: {e}")
            # Change back to original directory
            os.chdir(cwd) if 'cwd' in locals() else None
            return False
        except Exception as e:
            logger.error(f"Error pushing to repository: {e}")
            # Change back to original directory
            os.chdir(cwd) if 'cwd' in locals() else None
            return False

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create GitHub client
    client = GitHubClient()
    
    # Get accounts
    accounts = client.get_accounts()
    print(f"GitHub accounts: {accounts}")
    
    # Test connection if accounts exist
    if accounts:
        active_account = client.get_active_account()
        if active_account:
            print(f"Testing connection for {active_account}")
            if client.test_connection():
                print("Connection successful")
                
                # Get repositories
                repos = client.get_repositories()
                print(f"Repositories: {[repo['name'] for repo in repos]}")
            else:
                print("Connection failed")