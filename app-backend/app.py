import ollama
import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from utils.trash_classifyer import ImageAnalysis
from typing import Optional
from pydantic import BaseModel
import dotenv
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
# MODEL = "qwen3.5"
MODEL = os.getenv("MODEL")


# 1. Schema matching your exact requested categories
class ImageClassifier(BaseModel):
    organics: bool
    electronics: bool
    plastic: bool
    battery: bool

@app.post("/classify")
async def classify_image(
    file: Optional[UploadFile] = File(None), 
    path: Optional[str] = Form(None)
):
    image_source = None

    if file:
        image_source = await file.read()
    elif path:
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Local path not found")
        image_source = path 
    else:
        raise HTTPException(status_code=400, detail="Provide either a file or a path")

    try:
        # 2. Added the classifier logic using structured output
        response = ollama.chat(
            model=MODEL,
            messages=[{
                'role': 'user', 
                'content': 'Determine if the image contains organics, electronics, plastic, or batteries.',
                'images': [image_source]
            }],
            format=ImageAnalysis.model_json_schema()
        )
        
        # 3. Return the validated JSON structure
        return ImageAnalysis.model_validate_json(response.message.content)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/describe")
async def describe_image(
    file: Optional[UploadFile] = File(None), 
    path: Optional[str] = Form(None)
):
    # Your original describe logic remains untouched here
    image_source = await file.read() if file else path
    response = ollama.generate(model=MODEL, prompt="Describe this image.", images=[image_source])
    return {"description": response['response']}