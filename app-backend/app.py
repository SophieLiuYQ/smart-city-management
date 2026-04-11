import ollama
import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import Optional

app = FastAPI()
MODEL = "llama3.2-vision"

@app.post("/describe")
async def describe_image(
    file: Optional[UploadFile] = File(None), 
    path: Optional[str] = Form(None)
):
    image_source = None

    # 1. Handle File Upload (Blob)
    if file:
        image_source = await file.read()
    
    # 2. Handle Local Path (Fallback)
    elif path:
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Local path not found")
        # Ollama's Python SDK can accept the path string directly
        image_source = path 
    
    else:
        raise HTTPException(status_code=400, detail="Provide either a file or a path")

    try:
        response = ollama.generate(
            model=MODEL,
            prompt="Describe this image.",
            images=[image_source]
        )
        return {"description": response['response']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))