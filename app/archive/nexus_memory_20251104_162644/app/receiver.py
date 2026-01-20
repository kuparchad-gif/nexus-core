# 📂 Path: /Systems/engine/memory_service/receiver.py

from fastapi import FastAPI, UploadFile, File
import shutil
import os
from uuid import uuid4

app  =  FastAPI()

UPLOAD_DIR  =  '/memory/uploads/'
os.makedirs(UPLOAD_DIR, exist_ok = True)

@app.post("/upload_memory/")
async def upload_memory(file: UploadFile  =  File(...)):
    file_id  =  str(uuid4())
    filename  =  f"{file_id}_{file.filename}"
    file_path  =  os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"message": "Memory received.", "file_id": file_id, "file_path": file_path}

# To launch:
# uvicorn receiver:app --host 0.0.0.0 --port 8081
