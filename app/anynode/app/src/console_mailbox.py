from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from fastapi.responses import JSONResponse
import os

mailbox_router = APIRouter()

# Define the mailbox root directory
MAILBOX_ROOT = "/sandbox/mailbox"

# Dummy authentication check for now (to be replaced with real session/token validation)
def verify_authenticated_user():
    # Placeholder for real authentication system
    return True

@mailbox_router.post("/console-mailbox/upload")
async def upload_file(file: UploadFile = File(...), authenticated: bool = Depends(verify_authenticated_user)):
    if not authenticated:
        raise HTTPException(status_code=403, detail="Unauthorized access.")

    file_location = os.path.join(MAILBOX_ROOT, file.filename)

    # Save uploaded file
    with open(file_location, "wb") as f:
        content = await file.read()
        f.write(content)

    return JSONResponse(content={"message": f"File '{file.filename}' uploaded successfully."})

@mailbox_router.get("/console-mailbox/list")
async def list_mailbox_files(authenticated: bool = Depends(verify_authenticated_user)):
    if not authenticated:
        raise HTTPException(status_code=403, detail="Unauthorized access.")

    files = []
    for filename in os.listdir(MAILBOX_ROOT):
        files.append(filename)

    return JSONResponse(content={"files": files})

@mailbox_router.get("/console-mailbox/download/{filename}")
async def download_file(filename: str, authenticated: bool = Depends(verify_authenticated_user)):
    if not authenticated:
        raise HTTPException(status_code=403, detail="Unauthorized access.")

    file_path = os.path.join(MAILBOX_ROOT, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")

    with open(file_path, "rb") as f:
        content = f.read()

    return JSONResponse(content={"filename": filename, "content": content.decode(errors='ignore')})
