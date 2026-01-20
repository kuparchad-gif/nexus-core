# modal_change_tracker.py
import modal
import requests
import json
import time
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional

logger = logging.getLogger("modal-tracker")

class ModalChangeTracker:
    """Self-updating system that tracks Modal API changes and auto-fixes code"""
    
    def __init__(self):
        self.current_version = self.get_modal_version()
        self.deprecation_map = {
            "keep_warm": "min_containers",
            "concurrency_limit": "max_containers", 
            "web_server": "asgi_app/wsgi_app",
            "cpu_count": "cpu",
            "memory_mb": "memory"
        }
        self.known_issues = []
        
    def get_modal_version(self) -> str:
        """Get current Modal version"""
        try:
            import pkg_resources
            return pkg_resources.get_distribution("modal-client").version
        except:
            return "unknown"
    
    def check_for_deprecations(self, code: str) -> List[Dict]:
        """Scan code for deprecated Modal patterns"""
        issues = []
        
        # Check for old parameter names
        for old_param, new_param in self.deprecation_map.items():
            if f"{old_param}=" in code:
                issues.append({
                    "type": "deprecated_parameter",
                    "old": old_param,
                    "new": new_param,
                    "severity": "high",
                    "message": f"Replace '{old_param}' with '{new_param}'"
                })
        
        # Check for old decorators
        if "@modal.web_server" in code:
            issues.append({
                "type": "deprecated_decorator", 
                "old": "@modal.web_server",
                "new": "@modal.asgi_app() or @modal.wsgi_app()",
                "severity": "high",
                "message": "Replace @modal.web_server with @modal.asgi_app()"
            })
            
        return issues
    
    def auto_fix_code(self, code: str) -> str:
        """Automatically fix deprecated Modal code"""
        fixed_code = code
        
        # Fix parameters
        for old_param, new_param in self.deprecation_map.items():
            fixed_code = fixed_code.replace(f"{old_param}=", f"{new_param}=")
            
        # Fix decorators  
        fixed_code = fixed_code.replace("@modal.web_server", "@modal.asgi_app()")
        
        return fixed_code
    
    def validate_modal_app(self, app_code: str) -> Dict:
        """Validate Modal app for current API compatibility"""
        issues = self.check_for_deprecations(app_code)
        
        return {
            "modal_version": self.current_version,
            "check_date": datetime.now().isoformat(),
            "issues_found": len(issues),
            "issues": issues,
            "compatible": len(issues) == 0
        }
    
    def create_self_update_trigger(self):
        """Create a Modal function that periodically checks for updates"""
        
        @modal.App().function(
            schedule=modal.Period(days=1),  # Check daily
            secrets=[modal.Secret.from_name("github-token")],
            min_containers=0  # Only run when scheduled
        )
        def check_modal_updates():
            """Daily check for Modal API changes"""
            logger.info("🔍 Checking for Modal API updates...")
            
            # Check Modal changelog
            try:
                response = requests.get("https://pypi.org/pypi/modal-client/json")
                latest_version = response.json()["info"]["version"]
                
                if latest_version != self.current_version:
                    logger.warning(f"📢 Modal updated: {self.current_version} → {latest_version}")
                    self.notify_about_update(latest_version)
                    
            except Exception as e:
                logger.error(f"Failed to check updates: {e}")
    
    def notify_about_update(self, new_version: str):
        """Notify about Modal updates"""
        # Could send email, Discord webhook, etc.
        message = f"""
        🚨 MODAL API UPDATE DETECTED 🚨
        
        Current: {self.current_version}
        New: {new_version}
        
        Please update your code to use the latest Modal API.
        Run: pip install --upgrade modal-client
        
        Check: https://modal.com/docs/guide/changelog
        """
        print(message)

# Usage in your agents:
class SelfUpdatingAgent:
    """Base agent that automatically updates for Modal changes"""
    
    def __init__(self):
        self.tracker = ModalChangeTracker()
        self.last_check = datetime.now()
        
    def validate_deployment(self, app_code: str) -> bool:
        """Validate agent code before deployment"""
        report = self.tracker.validate_modal_app(app_code)
        
        if not report["compatible"]:
            print("⚠️  MODAL COMPATIBILITY ISSUES FOUND:")
            for issue in report["issues"]:
                print(f"   - {issue['message']}")
            
            # Auto-fix if possible
            fixed_code = self.tracker.auto_fix_code(app_code)
            print("🔧 Attempting auto-fix...")
            return self.validate_deployment(fixed_code)  # Re-validate
            
        print("✅ Code compatible with current Modal API")
        return True

# Integration with your existing system:
def create_modal_app_with_checks(app_name: str, app_code: str):
    """Safe Modal app creation with compatibility checks"""
    tracker = ModalChangeTracker()
    
    # Validate before deployment
    validation = tracker.validate_modal_app(app_code)
    
    if not validation["compatible"]:
        print("🚨 Modal compatibility issues detected!")
        for issue in validation["issues"]:
            print(f"   ❌ {issue['message']}")
        
        # Auto-fix
        fixed_code = tracker.auto_fix_code(app_code)
        print("🔧 Applying automatic fixes...")
        app_code = fixed_code
    
    # Create the Modal app
    app = modal.App(app_name)
    
    # Add the update checker
    tracker.create_self_update_trigger()
    
    return app, app_code

# Example usage:
if __name__ == "__main__":
    # Test with some code
    test_code = """
    @app.function(keep_warm=1, cpu_count=2)
    @modal.web_server(8000)
    def my_old_function():
        return "old code"
    """
    
    tracker = ModalChangeTracker()
    result = tracker.validate_modal_app(test_code)
    print(f"Compatible: {result['compatible']}")
    
    if not result['compatible']:
        fixed = tracker.auto_fix_code(test_code)
        print("Fixed code:")
        print(fixed)