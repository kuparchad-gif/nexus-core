from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
import os

viewer_router  =  APIRouter()

# Define the root directory for allowed file viewing (sandboxed)
VIEWER_ROOT  =  "/sandbox/files"  # Adjust this to wherever you store safe viewable files

# Dummy authentication check for now (to be replaced with real session/token validation)
def verify_authenticated_user():
    # Placeholder for real authentication system
    return True

@viewer_router.get("/console-viewer")
async def list_viewable_files(authenticated: bool  =  Depends(verify_authenticated_user)):
    if not authenticated:
        raise HTTPException(status_code = 403, detail = "Unauthorized access.")

    files  =  []
    for root, dirs, filenames in os.walk(VIEWER_ROOT):
        for filename in filenames:
            filepath  =  os.path.join(root, filename)
            relative_path  =  os.path.relpath(filepath, VIEWER_ROOT)
            files.append(relative_path)

    return JSONResponse(content = {"files": files})
