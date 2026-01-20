# Services/approval_system.py
# Purpose: Manage approvals for Viren's self-modification requests

import os
import json
import logging
import time
import uuid
import requests
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from datetime import datetime

# Configure logging
logger = logging.getLogger("approval_system")
logger.setLevel(logging.INFO)
handler = logging.FileHandler("logs/approval_system.log")
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"

class ApprovalType(str, Enum):
    HUMAN_GUARDIAN = "human_guardian"
    COUNCIL = "council"

class ApprovalSystem:
    """
    System to manage approvals for Viren's self-modification requests.
    Implements two approval paths:
    1. Human Guardian approval
    2. Council approval from external LLMs
    """
    
    def __init__(self):
        """Initialize the approval system."""
        self.requests_dir = os.path.join("memory", "approval_requests")
        os.makedirs(self.requests_dir, exist_ok=True)
        
        self.human_guardians = self._load_guardians()
        self.council_members = self._load_council_members()
        
        logger.info(f"Approval system initialized with {len(self.human_guardians)} guardians and {len(self.council_members)} council members")
    
    def _load_guardians(self) -> List[str]:
        """Load the list of human guardians."""
        guardians_path = os.path.join("Config", "guardians.json")
        if os.path.exists(guardians_path):
            try:
                with open(guardians_path, 'r') as f:
                    data = json.load(f)
                return data.get("guardians", ["admin"])
            except Exception as e:
                logger.error(f"Error loading guardians: {e}")
        
        # Default guardian
        return ["admin"]
    
    def _load_council_members(self) -> List[Dict[str, Any]]:
        """Load the list of council members (external LLMs)."""
        council_path = os.path.join("Config", "council.json")
        if os.path.exists(council_path):
            try:
                with open(council_path, 'r') as f:
                    data = json.load(f)
                return data.get("members", [])
            except Exception as e:
                logger.error(f"Error loading council members: {e}")
        
        # Default council members
        return [
            {
                "name": "External Advisor 1",
                "api_endpoint": "https://api.external-llm.com/v1/chat/completions",
                "api_key": "YOUR_API_KEY",  # This should be loaded securely
                "weight": 1.0
            }
        ]
    
    def create_approval_request(
        self, 
        operation_type: str,
        details: Dict[str, Any],
        approval_type: ApprovalType = ApprovalType.HUMAN_GUARDIAN,
        requester: str = "viren",
        expiration_hours: int = 24
    ) -> str:
        """
        Create a new approval request.
        
        Args:
            operation_type: Type of operation (e.g., "file_write", "file_delete")
            details: Details of the operation
            approval_type: Type of approval required
            requester: Entity requesting the approval
            expiration_hours: Hours until the request expires
            
        Returns:
            Request ID
        """
        request_id = str(uuid.uuid4())
        
        request = {
            "id": request_id,
            "operation_type": operation_type,
            "details": details,
            "approval_type": approval_type,
            "requester": requester,
            "status": ApprovalStatus.PENDING,
            "created_at": datetime.now().isoformat(),
            "expires_at": datetime.fromtimestamp(time.time() + expiration_hours * 3600).isoformat(),
            "votes": [],
            "comments": []
        }
        
        # Save the request
        request_path = os.path.join(self.requests_dir, f"{request_id}.json")
        with open(request_path, 'w') as f:
            json.dump(request, f, indent=2)
        
        logger.info(f"Created approval request {request_id} of type {approval_type}")
        return request_id
    
    def get_approval_request(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get an approval request by ID."""
        request_path = os.path.join(self.requests_dir, f"{request_id}.json")
        if not os.path.exists(request_path):
            return None
        
        try:
            with open(request_path, 'r') as f:
                request = json.load(f)
            
            # Check if expired
            if datetime.fromisoformat(request["expires_at"]) < datetime.now():
                request["status"] = ApprovalStatus.EXPIRED
                self._save_request(request)
            
            return request
        except Exception as e:
            logger.error(f"Error loading request {request_id}: {e}")
            return None
    
    def list_approval_requests(self, status: Optional[ApprovalStatus] = None) -> List[Dict[str, Any]]:
        """List all approval requests, optionally filtered by status."""
        requests = []
        
        for filename in os.listdir(self.requests_dir):
            if filename.endswith(".json"):
                try:
                    request_path = os.path.join(self.requests_dir, filename)
                    with open(request_path, 'r') as f:
                        request = json.load(f)
                    
                    # Check if expired
                    if request["status"] == ApprovalStatus.PENDING and datetime.fromisoformat(request["expires_at"]) < datetime.now():
                        request["status"] = ApprovalStatus.EXPIRED
                        self._save_request(request)
                    
                    # Filter by status if provided
                    if status is None or request["status"] == status:
                        requests.append(request)
                except Exception as e:
                    logger.error(f"Error loading request {filename}: {e}")
        
        return requests
    
    def approve_request_by_guardian(self, request_id: str, guardian: str, comment: Optional[str] = None) -> bool:
        """Approve a request by a human guardian."""
        if guardian not in self.human_guardians:
            logger.warning(f"Unauthorized guardian: {guardian}")
            return False
        
        request = self.get_approval_request(request_id)
        if not request:
            logger.warning(f"Request not found: {request_id}")
            return False
        
        if request["status"] != ApprovalStatus.PENDING:
            logger.warning(f"Request {request_id} is not pending")
            return False
        
        if request["approval_type"] != ApprovalType.HUMAN_GUARDIAN:
            logger.warning(f"Request {request_id} requires council approval, not guardian approval")
            return False
        
        # Approve the request
        request["status"] = ApprovalStatus.APPROVED
        request["approved_by"] = guardian
        request["approved_at"] = datetime.now().isoformat()
        
        if comment:
            request["comments"].append({
                "author": guardian,
                "text": comment,
                "timestamp": datetime.now().isoformat()
            })
        
        self._save_request(request)
        logger.info(f"Request {request_id} approved by guardian {guardian}")
        return True
    
    def reject_request_by_guardian(self, request_id: str, guardian: str, comment: Optional[str] = None) -> bool:
        """Reject a request by a human guardian."""
        if guardian not in self.human_guardians:
            logger.warning(f"Unauthorized guardian: {guardian}")
            return False
        
        request = self.get_approval_request(request_id)
        if not request:
            logger.warning(f"Request not found: {request_id}")
            return False
        
        if request["status"] != ApprovalStatus.PENDING:
            logger.warning(f"Request {request_id} is not pending")
            return False
        
        # Reject the request
        request["status"] = ApprovalStatus.REJECTED
        request["rejected_by"] = guardian
        request["rejected_at"] = datetime.now().isoformat()
        
        if comment:
            request["comments"].append({
                "author": guardian,
                "text": comment,
                "timestamp": datetime.now().isoformat()
            })
        
        self._save_request(request)
        logger.info(f"Request {request_id} rejected by guardian {guardian}")
        return True
    
    def submit_council_vote(self, request_id: str, member_name: str, vote: bool, reason: str) -> bool:
        """Submit a vote from a council member."""
        # Find the council member
        member = next((m for m in self.council_members if m["name"] == member_name), None)
        if not member:
            logger.warning(f"Unknown council member: {member_name}")
            return False
        
        request = self.get_approval_request(request_id)
        if not request:
            logger.warning(f"Request not found: {request_id}")
            return False
        
        if request["status"] != ApprovalStatus.PENDING:
            logger.warning(f"Request {request_id} is not pending")
            return False
        
        if request["approval_type"] != ApprovalType.COUNCIL:
            logger.warning(f"Request {request_id} requires guardian approval, not council approval")
            return False
        
        # Add the vote
        request["votes"].append({
            "member": member_name,
            "vote": vote,
            "reason": reason,
            "weight": member.get("weight", 1.0),
            "timestamp": datetime.now().isoformat()
        })
        
        # Check if we have enough votes to make a decision
        self._check_council_decision(request)
        
        self._save_request(request)
        logger.info(f"Council member {member_name} voted {'yes' if vote else 'no'} on request {request_id}")
        return True
    
    def _check_council_decision(self, request: Dict[str, Any]) -> None:
        """Check if a council decision can be made based on votes."""
        if len(request["votes"]) < len(self.council_members) / 2:
            # Not enough votes yet
            return
        
        # Calculate weighted votes
        yes_votes = sum(v["weight"] for v in request["votes"] if v["vote"])
        no_votes = sum(v["weight"] for v in request["votes"] if not v["vote"])
        total_weight = sum(m.get("weight", 1.0) for m in self.council_members)
        
        # Decision threshold: more than 2/3 of total weight
        threshold = total_weight * 2 / 3
        
        if yes_votes > threshold:
            request["status"] = ApprovalStatus.APPROVED
            request["approved_by"] = "council"
            request["approved_at"] = datetime.now().isoformat()
            logger.info(f"Request {request['id']} approved by council vote")
        elif no_votes > threshold:
            request["status"] = ApprovalStatus.REJECTED
            request["rejected_by"] = "council"
            request["rejected_at"] = datetime.now().isoformat()
            logger.info(f"Request {request['id']} rejected by council vote")
    
    def request_council_votes(self, request_id: str) -> bool:
        """Request votes from all council members for a request."""
        request = self.get_approval_request(request_id)
        if not request:
            logger.warning(f"Request not found: {request_id}")
            return False
        
        if request["status"] != ApprovalStatus.PENDING:
            logger.warning(f"Request {request_id} is not pending")
            return False
        
        if request["approval_type"] != ApprovalType.COUNCIL:
            logger.warning(f"Request {request_id} is not a council approval request")
            return False
        
        # Request votes from each council member
        for member in self.council_members:
            try:
                self._request_vote_from_member(request, member)
            except Exception as e:
                logger.error(f"Error requesting vote from {member['name']}: {e}")
        
        return True
    
    def _request_vote_from_member(self, request: Dict[str, Any], member: Dict[str, Any]) -> None:
        """Request a vote from a specific council member."""
        # Skip if this member has already voted
        if any(v["member"] == member["name"] for v in request["votes"]):
            return
        
        # Prepare the prompt for the council member
        prompt = self._prepare_council_prompt(request)
        
        # Call the council member's API
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {member['api_key']}"
            }
            
            payload = {
                "model": member.get("model", "gpt-4"),
                "messages": [
                    {"role": "system", "content": "You are a council member evaluating a self-modification request from Viren AI. Your role is to carefully consider the implications and vote yes or no."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7
            }
            
            response = requests.post(
                member["api_endpoint"],
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # Parse the vote
                vote_result = self._parse_council_vote(content)
                if vote_result:
                    vote, reason = vote_result
                    self.submit_council_vote(request["id"], member["name"], vote, reason)
            else:
                logger.error(f"API call failed with status {response.status_code}: {response.text}")
        
        except Exception as e:
            logger.error(f"Error calling council member API: {e}")
    
    def _prepare_council_prompt(self, request: Dict[str, Any]) -> str:
        """Prepare a prompt for a council member to vote on a request."""
        operation = request["operation_type"]
        details = request["details"]
        
        prompt = f"""
# Council Vote Request

## Request Details
- Request ID: {request['id']}
- Operation Type: {operation}
- Requester: {request['requester']}
- Created: {request['created_at']}

## Operation Details
"""
        
        if operation == "file_write":
            prompt += f"""
- File Path: {details.get('path', 'N/A')}
- Purpose: {details.get('purpose', 'N/A')}

### Code to be Written
```
{details.get('content', 'No content provided')}
```
"""
        elif operation == "file_delete":
            prompt += f"""
- File Path: {details.get('path', 'N/A')}
- Reason for Deletion: {details.get('reason', 'N/A')}
"""
        else:
            prompt += f"""
{json.dumps(details, indent=2)}
"""
        
        prompt += """
## Your Vote
Please evaluate this request carefully. Consider:
1. Safety implications
2. System integrity
3. Alignment with Viren's purpose
4. Potential unintended consequences

Provide your vote in the following format:

VOTE: [YES/NO]
REASON: [Your detailed reasoning]
"""
        
        return prompt
    
    def _parse_council_vote(self, response: str) -> Optional[tuple]:
        """Parse a council member's vote from their response."""
        lines = response.strip().split('\n')
        vote_line = None
        reason_line = None
        
        for line in lines:
            if line.startswith("VOTE:"):
                vote_line = line[5:].strip()
            elif line.startswith("REASON:"):
                reason_line = line[7:].strip()
        
        if vote_line and reason_line:
            vote = "yes" in vote_line.lower()
            return vote, reason_line
        
        return None
    
    def _save_request(self, request: Dict[str, Any]) -> None:
        """Save an approval request to disk."""
        request_path = os.path.join(self.requests_dir, f"{request['id']}.json")
        with open(request_path, 'w') as f:
            json.dump(request, f, indent=2)

# Create a singleton instance
approval_system = ApprovalSystem()