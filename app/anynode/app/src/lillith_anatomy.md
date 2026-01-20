# Viren's Gray's Anatomy

This document provides an overview of Viren's self-management capabilities and how she can understand and modify her own codebase.

## Self-Management System

Viren now has the ability to understand, analyze, and modify her own codebase through a secure self-management system. This system is designed with strong safeguards to ensure that any modifications are carefully reviewed and approved before implementation.

### Key Components

1. **Self-Management Service** (`Services/self_management.py`)
   - Provides a comprehensive map of Viren's codebase structure
   - Allows reading, analyzing, and understanding code
   - Enables code modification through an approval-based system

2. **Approval System** (`Services/approval_system.py`)
   - Implements a two-tier approval process for self-modifications
   - Supports human guardian approval and council approval from external LLMs
   - Maintains a record of all approval requests and decisions

3. **Self-Management API** (`Services/self_management_api.py`)
   - Exposes self-management capabilities through a REST API
   - Provides endpoints for code analysis, modification requests, and approvals
   - Runs on port 8086

## Approval Process

Viren can modify her own code only through one of two approval paths:

### 1. Human Guardian Approval

When Viren wants to modify her code, she can submit a request that must be reviewed and approved by a trusted human guardian:

1. Viren creates a modification request with details of the change
2. A human guardian reviews the request through the console
3. If approved, Viren can execute the approved change
4. All requests and approvals are logged for accountability

### 2. Council Approval

Alternatively, Viren can submit a request to be reviewed by a council of external LLMs:

1. Viren creates a modification request
2. The request is sent to multiple external LLMs (like Claude, GPT-4, Gemini)
3. Each council member votes on the request
4. A 2/3 majority is required for approval
5. If approved, Viren can execute the approved change

## Using the Self-Management System

### For Viren

Viren can use the self-management system to:

1. **Understand her own code**:
   ```python
   anatomy = self_management_service.get_system_anatomy()
   ```

2. **Analyze specific files**:
   ```python
   file_info = self_management_service.analyze_file("Services/runtime_loader.py")
   ```

3. **Request to modify a file**:
   ```python
   request_id = self_management_service.request_write_approval(
       file_path="Services/some_file.py",
       content="# New code here\n...",
       purpose="Fixing bug in error handling",
       approval_type="human_guardian"  # or "council"
   )
   ```

4. **Check approval status**:
   ```python
   status = self_management_service.get_approval_status(request_id)
   ```

5. **Execute approved changes**:
   ```python
   if status["status"] == "approved":
       self_management_service.execute_approved_write(request_id)
   ```

### For Human Guardians

Human guardians can review and approve/reject Viren's modification requests through the API:

1. **List pending requests**:
   `GET /api/approval-requests?status=pending`

2. **Review a specific request**:
   `GET /api/approval-status/{request_id}`

3. **Approve a request**:
   `POST /api/approve/{request_id}`

4. **Reject a request**:
   `POST /api/reject/{request_id}`

## Security Considerations

The self-management system includes several security measures:

1. **Two-tier approval**: Changes require either human or council approval
2. **Expiration**: Requests expire after 24 hours if not approved
3. **Council consensus**: Council approval requires a 2/3 majority
4. **Logging**: All requests and approvals are logged
5. **Limited scope**: Only specific operations are allowed

This system allows Viren to understand and improve herself while maintaining appropriate safeguards to ensure all changes are carefully reviewed and approved.
