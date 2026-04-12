import ollama
from pydantic import BaseModel
from typing import List
from typing import Optional, List, Literal

# 1. Define the classification schema
class WasteClassification(BaseModel):
    contains_organics: bool
    contains_electronics: bool
    contains_plastic: bool
    contains_batteries: bool
    confidence_score: float
    detected_items: List[str]

class ImageAnalysis(BaseModel):
    detected_materials: List[Literal["organics", "electronics", "plastic", "battery"]]

# 2. Request structured output from a vision model
# Ensure you have the vision model pulled (e.g., ollama pull qwen3.5)
IMAGE_PATH = 'path_to_your_image.jpg'

response = ollama.chat(
    model='qwen3.5',
    messages=[{
        'role': 'user',
        'content': 'Analyze this image and identify if it contains organics, electronics, plastic, or batteries.',
        'images': [IMAGE_PATH]
    }],
    format=WasteClassification.model_json_schema()
)

# 3. Parse and use the structured data
analysis = WasteClassification.model_validate_json(response.message.content)

print(f"Analysis Results for {IMAGE_PATH}:")
print(f"- Organics: {analysis.contains_organics}")
print(f"- Electronics: {analysis.contains_electronics}")
print(f"- Plastic: {analysis.contains_plastic}")
print(f"- Batteries: {analysis.contains_batteries}")
print(f"- Detected Items: {', '.join(analysis.detected_items)}")