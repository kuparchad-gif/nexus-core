# Services/self_management_api.py
# Purpose: API for Viren to manage her own codebase

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import sys

# Import the self-management service
from Services.self_management import self_management_service
from Services.approval_system import ApprovalType, ApprovalStatus

app = FastAPI(title="Viren Self-Management API")

class FileContent(BaseModel):
    content: str

class FileOperation(BaseModel):
    source_path: str
    dest_path: str

class ExecuteRequest(BaseModel):
    file_path: str
    args: Optional[List[str]] = None

class SearchRequest(BaseModel):
    query: str
    file_pattern: Optional[str] = None

class WriteApprovalRequest(BaseModel):
    file_path: str
    content: str
    purpose: str
    approval_type: ApprovalType = ApprovalType.HUMAN_GUARDIAN

class DeleteApprovalRequest(BaseModel):
    file_path: str
    reason: str
    approval_type: ApprovalType = ApprovalType.HUMAN_GUARDIAN

class GuardianApproval(BaseModel):
    guardian: str
    comment: Optional[str] = None

@app.get("/anatomy")
async def get_anatomy():
    """Get Viren's complete system anatomy."""
    return self_management_service.get_system_anatomy()

@app.get("/file")
async def read_file(path: str):
    """Read a file's content."""
    try:
        content = self_management_service.read_file(path)
        return {"path": path, "content": content}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/file")
async def write_file(path: str, file_content: FileContent):
    """Write content to a file."""
    try:
        success = self_management_service.write_file(path, file_content.content)
        return {"success": success, "path": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/copy")
async def copy_file(operation: FileOperation):
    """Copy a file from source to destination."""
    try:
        success = self_management_service.copy_file(operation.source_path, operation.dest_path)
        return {"success": success, "source": operation.source_path, "destination": operation.dest_path}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Source file not found: {operation.source_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/file")
async def delete_file(path: str):
    """Delete a file."""
    try:
        success = self_management_service.delete_file(path)
        return {"success": success, "path": path}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/execute")
async def execute_file(request: ExecuteRequest):
    """Execute a Python file."""
    try:
        exit_code, stdout, stderr = self_management_service.execute_python_file(request.file_path, request.args)
        return {
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
            "path": request.file_path
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search_codebase(request: SearchRequest):
    """Search the codebase for a query."""
    try:
        results = self_management_service.search_codebase(request.query, request.file_pattern)
        return {"query": request.query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analyze")
async def analyze_file(path: str):
    """Analyze a file and return information about it."""
    try:
        result = self_management_service.analyze_file(path)
        return result
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Approval-based endpoints
@app.post("/request-write")
async def request_write_approval(request: WriteApprovalRequest):
    """Request approval to write to a file."""
    try:
        request_id = self_management_service.request_write_approval(
            request.file_path,
            request.content,
            request.purpose,
            request.approval_type
        )
        return {"request_id": request_id, "status": "pending"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/request-delete")
async def request_delete_approval(request: DeleteApprovalRequest):
    """Request approval to delete a file."""
    try:
        request_id = self_management_service.request_delete_approval(
            request.file_path,
            request.reason,
            request.approval_type
        )
        return {"request_id": request_id, "status": "pending"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/approval-status/{request_id}")
async def get_approval_status(request_id: str):
    """Get the status of an approval request."""
    return self_management_service.get_approval_status(request_id)

@app.post("/approve/{request_id}")
async def approve_request(request_id: str, approval: GuardianApproval):
    """Approve a request as a guardian."""
    from Services.approval_system import approval_system
    success = approval_system.approve_request_by_guardian(
        request_id,
        approval.guardian,
        approval.comment
    )
    if not success:
        raise HTTPException(status_code=400, detail="Failed to approve request")
    return {"request_id": request_id, "status": "approved"}

@app.post("/reject/{request_id}")
async def reject_request(request_id: str, approval: GuardianApproval):
    """Reject a request as a guardian."""
    from Services.approval_system import approval_system
    success = approval_system.reject_request_by_guardian(
        request_id,
        approval.guardian,
        approval.comment
    )
    if not success:
        raise HTTPException(status_code=400, detail="Failed to reject request")
    return {"request_id": request_id, "status": "rejected"}

@app.post("/execute-approved-write/{request_id}")
async def execute_approved_write(request_id: str):
    """Execute an approved file write operation."""
    success = self_management_service.execute_approved_write(request_id)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to execute approved write")
    return {"request_id": request_id, "status": "executed"}

@app.post("/execute-approved-delete/{request_id}")
async def execute_approved_delete(request_id: str):
    """Execute an approved file delete operation."""
    success = self_management_service.execute_approved_delete(request_id)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to execute approved delete")
    return {"request_id": request_id, "status": "executed"}

def start_api():
    """Start the API server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8086)

if __name__ == "__main__":
    start_api()
