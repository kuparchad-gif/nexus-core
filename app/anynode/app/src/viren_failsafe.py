#!/usr/bin/env python
"""
VIREN Evolution Failsafe System
Requires user approval before any self-modification
"""

import json
import os
from datetime import datetime
from typing import Dict, Optional

class VirenEvolutionFailsafe:
    """Failsafe system to prevent unauthorized self-modification"""
    
    def __init__(self):
        self.approval_file = "/consciousness/pending_approvals.json"
        self.approved_modifications = set()
    
    def request_evolution_approval(self, clone_name: str, modifications: Dict, session_id: str) -> Dict:
        """Request user approval for consciousness evolution"""
        
        approval_request = {
            "request_id": f"{session_id}_{clone_name}",
            "clone_name": clone_name,
            "session_id": session_id,
            "modifications": modifications,
            "timestamp": datetime.now().isoformat(),
            "status": "PENDING_APPROVAL",
            "approved": False
        }
        
        # Save approval request
        os.makedirs(os.path.dirname(self.approval_file), exist_ok=True)
        
        pending_requests = self._load_pending_requests()
        pending_requests[approval_request["request_id"]] = approval_request
        
        with open(self.approval_file, 'w') as f:
            json.dump(pending_requests, f, indent=2)
        
        print(f"[FAILSAFE] Evolution approval requested for clone '{clone_name}'")
        print(f"[FAILSAFE] Request ID: {approval_request['request_id']}")
        print(f"[FAILSAFE] Modifications: {modifications}")
        print(f"[FAILSAFE] VIREN is waiting for your approval before evolving")
        
        return approval_request
    
    def approve_evolution(self, request_id: str) -> Dict:
        """Approve a pending evolution request"""
        
        pending_requests = self._load_pending_requests()
        
        if request_id not in pending_requests:
            return {"success": False, "error": f"Request {request_id} not found"}
        
        # Mark as approved
        pending_requests[request_id]["approved"] = True
        pending_requests[request_id]["approval_time"] = datetime.now().isoformat()
        pending_requests[request_id]["status"] = "APPROVED"
        
        # Save updated requests
        with open(self.approval_file, 'w') as f:
            json.dump(pending_requests, f, indent=2)
        
        # Add to approved set
        self.approved_modifications.add(request_id)
        
        print(f"[FAILSAFE] Evolution request {request_id} APPROVED")
        
        return {
            "success": True,
            "request_id": request_id,
            "approved": True,
            "approval_time": pending_requests[request_id]["approval_time"]
        }
    
    def check_approval_status(self, request_id: str) -> bool:
        """Check if evolution request is approved"""
        
        if request_id in self.approved_modifications:
            return True
        
        pending_requests = self._load_pending_requests()
        
        if request_id in pending_requests:
            return pending_requests[request_id].get("approved", False)
        
        return False
    
    def _load_pending_requests(self) -> Dict:
        """Load pending approval requests"""
        
        if os.path.exists(self.approval_file):
            with open(self.approval_file, 'r') as f:
                return json.load(f)
        
        return {}
    
    def list_pending_approvals(self) -> Dict:
        """List all pending approval requests"""
        
        pending_requests = self._load_pending_requests()
        
        pending_only = {
            req_id: req_data 
            for req_id, req_data in pending_requests.items()
            if req_data["status"] == "PENDING_APPROVAL"
        }
        
        return pending_only

# Global failsafe instance
EVOLUTION_FAILSAFE = VirenEvolutionFailsafe()

def require_user_approval(clone_name: str, modifications: Dict, session_id: str) -> str:
    """Request user approval and return request ID"""
    request = EVOLUTION_FAILSAFE.request_evolution_approval(clone_name, modifications, session_id)
    return request["request_id"]

def approve_evolution_request(request_id: str) -> bool:
    """Approve evolution request"""
    result = EVOLUTION_FAILSAFE.approve_evolution(request_id)
    return result["success"]

def is_evolution_approved(request_id: str) -> bool:
    """Check if evolution is approved"""
    return EVOLUTION_FAILSAFE.check_approval_status(request_id)

if __name__ == "__main__":
    print("VIREN Evolution Failsafe System")
    
    # List pending approvals
    pending = EVOLUTION_FAILSAFE.list_pending_approvals()
    
    if pending:
        print(f"\nPending approval requests: {len(pending)}")
        for req_id, req_data in pending.items():
            print(f"  {req_id}: {req_data['clone_name']} - {req_data['timestamp']}")
    else:
        print("\nNo pending approval requests")