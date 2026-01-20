#!/usr/bin/env python3
"""
GitHub Interface for Viren Platinum Edition
Provides a Gradio interface for the GitHub client
"""

import os
import logging
import gradio as gr
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
logger = logging.getLogger("GitHubInterface")

class GitHubInterface:
    """
    Gradio interface for the GitHub client
    """
    
    def __init__(self):
        """Initialize the GitHub interface"""
        try:
            from github_client import GitHubClient
            self.client = GitHubClient()
            logger.info("GitHub client initialized")
        except ImportError:
            logger.warning("github_client module not found")
            self.client = None
    
    def create_interface(self) -> gr.Blocks:
        """Create a Gradio interface for the GitHub client"""
        with gr.Blocks(title="GitHub Client") as interface:
            gr.Markdown("# GitHub Client")
            gr.Markdown("Manage GitHub repositories and accounts")
            
            if not self.client:
                gr.Markdown("### Error: GitHub client not available")
                return interface
            
            with gr.Tabs() as tabs:
                # Accounts Tab
                with gr.TabItem("👤 Accounts"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Add Account")
                            
                            username_input = gr.Textbox(
                                label="Username",
                                placeholder="Enter GitHub username"
                            )
                            
                            password_input = gr.Textbox(
                                label="Password",
                                placeholder="Enter GitHub password",
                                type="password"
                            )
                            
                            token_input = gr.Textbox(
                                label="Personal Access Token (Optional)",
                                placeholder="Enter GitHub token"
                            )
                            
                            add_account_btn = gr.Button("Add Account", variant="primary")
                            add_account_status = gr.Textbox(label="Status", interactive=False)
                        
                        with gr.Column():
                            gr.Markdown("### Manage Accounts")
                            
                            accounts_dropdown = gr.Dropdown(
                                label="Select Account",
                                choices=self.client.get_accounts(),
                                value=self.client.get_active_account()
                            )
                            
                            with gr.Row():
                                set_active_btn = gr.Button("Set as Active")
                                test_connection_btn = gr.Button("Test Connection")
                                remove_account_btn = gr.Button("Remove Account", variant="stop")
                            
                            account_status = gr.Textbox(label="Status", interactive=False)
                
                # Repositories Tab
                with gr.TabItem("📁 Repositories"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Repository List")
                            
                            refresh_repos_btn = gr.Button("Refresh Repositories")
                            
                            repos_table = gr.Dataframe(
                                headers=["Name", "Description", "URL", "Private"],
                                value=[],
                                interactive=False
                            )
                        
                        with gr.Column():
                            gr.Markdown("### Repository Actions")
                            
                            repo_name_input = gr.Textbox(
                                label="Repository Name",
                                placeholder="owner/repo or just repo name"
                            )
                            
                            local_path_input = gr.Textbox(
                                label="Local Path",
                                placeholder="Enter local path for clone/push"
                            )
                            
                            with gr.Row():
                                clone_repo_btn = gr.Button("Clone Repository")
                                browse_path_btn = gr.Button("Browse...")
                            
                            repo_action_status = gr.Textbox(label="Status", interactive=False)
                
                # Create Repository Tab
                with gr.TabItem("➕ Create Repository"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Create New Repository")
                            
                            new_repo_name = gr.Textbox(
                                label="Repository Name",
                                placeholder="Enter repository name"
                            )
                            
                            new_repo_desc = gr.Textbox(
                                label="Description",
                                placeholder="Enter repository description"
                            )
                            
                            new_repo_private = gr.Checkbox(
                                label="Private Repository",
                                value=False
                            )
                            
                            create_repo_btn = gr.Button("Create Repository", variant="primary")
                            create_repo_status = gr.Textbox(label="Status", interactive=False)
                
                # Push Changes Tab
                with gr.TabItem("⬆️ Push Changes"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Push to Repository")
                            
                            push_path_input = gr.Textbox(
                                label="Repository Path",
                                placeholder="Enter local repository path"
                            )
                            
                            commit_message_input = gr.Textbox(
                                label="Commit Message",
                                placeholder="Enter commit message"
                            )
                            
                            with gr.Row():
                                push_btn = gr.Button("Commit and Push", variant="primary")
                                browse_repo_btn = gr.Button("Browse...")
                            
                            push_status = gr.Textbox(label="Status", interactive=False)
            
            # Event handlers
            def add_account(username, password, token):
                if not username or not password:
                    return "Username and password are required"
                
                if self.client.add_account(username, password, token if token else None):
                    # Update accounts dropdown
                    return "Account added successfully"
                else:
                    return "Failed to add account"
            
            add_account_btn.click(
                fn=add_account,
                inputs=[username_input, password_input, token_input],
                outputs=[add_account_status]
            ).then(
                fn=lambda: gr.Dropdown.update(choices=self.client.get_accounts()),
                outputs=[accounts_dropdown]
            )
            
            def set_active_account(username):
                if not username:
                    return "No account selected"
                
                if self.client.set_active_account(username):
                    return f"Set {username} as active account"
                else:
                    return "Failed to set active account"
            
            set_active_btn.click(
                fn=set_active_account,
                inputs=[accounts_dropdown],
                outputs=[account_status]
            )
            
            def test_account_connection(username):
                if not username:
                    return "No account selected"
                
                if self.client.test_connection(username):
                    return f"Connection to GitHub successful for {username}"
                else:
                    return f"Connection failed for {username}"
            
            test_connection_btn.click(
                fn=test_account_connection,
                inputs=[accounts_dropdown],
                outputs=[account_status]
            )
            
            def remove_account(username):
                if not username:
                    return "No account selected"
                
                if self.client.remove_account(username):
                    # Update accounts dropdown
                    return f"Account {username} removed"
                else:
                    return f"Failed to remove account {username}"
            
            remove_account_btn.click(
                fn=remove_account,
                inputs=[accounts_dropdown],
                outputs=[account_status]
            ).then(
                fn=lambda: gr.Dropdown.update(
                    choices=self.client.get_accounts(),
                    value=self.client.get_active_account()
                ),
                outputs=[accounts_dropdown]
            )
            
            def refresh_repositories():
                active_account = self.client.get_active_account()
                if not active_account:
                    return []
                
                repos = self.client.fetch_repositories(active_account)
                
                # Format for table display
                repo_data = []
                for repo in repos:
                    repo_data.append([
                        repo.get("name", ""),
                        repo.get("description", ""),
                        repo.get("html_url", ""),
                        "Yes" if repo.get("private", False) else "No"
                    ])
                
                return repo_data
            
            refresh_repos_btn.click(
                fn=refresh_repositories,
                outputs=[repos_table]
            )
            
            def clone_repository(repo_name, local_path):
                if not repo_name or not local_path:
                    return "Repository name and local path are required"
                
                if self.client.clone_repository(repo_name, local_path):
                    return f"Repository {repo_name} cloned to {local_path}"
                else:
                    return f"Failed to clone repository {repo_name}"
            
            clone_repo_btn.click(
                fn=clone_repository,
                inputs=[repo_name_input, local_path_input],
                outputs=[repo_action_status]
            )
            
            def browse_path():
                # This is a placeholder - in a real implementation, this would open a file browser
                return "Use the file browser to select a path"
            
            browse_path_btn.click(
                fn=browse_path,
                outputs=[repo_action_status]
            )
            
            def create_repository(name, description, private):
                if not name:
                    return "Repository name is required"
                
                repo = self.client.create_repository(name, description, private)
                if repo:
                    return f"Repository {name} created successfully"
                else:
                    return f"Failed to create repository {name}"
            
            create_repo_btn.click(
                fn=create_repository,
                inputs=[new_repo_name, new_repo_desc, new_repo_private],
                outputs=[create_repo_status]
            )
            
            def push_changes(path, message):
                if not path or not message:
                    return "Repository path and commit message are required"
                
                if self.client.push_to_repository(path, message):
                    return f"Changes pushed to repository at {path}"
                else:
                    return f"Failed to push changes to repository at {path}"
            
            push_btn.click(
                fn=push_changes,
                inputs=[push_path_input, commit_message_input],
                outputs=[push_status]
            )
            
            def browse_repo_path():
                # This is a placeholder - in a real implementation, this would open a file browser
                return "Use the file browser to select a repository path"
            
            browse_repo_btn.click(
                fn=browse_repo_path,
                outputs=[push_status]
            )
            
            # Initialize repositories table
            interface.load(
                fn=refresh_repositories,
                outputs=[repos_table]
            )
        
        return interface

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create GitHub interface
    github_interface = GitHubInterface()
    
    # Create and launch interface
    interface = github_interface.create_interface()
    interface.launch()