# 🖼️ FastAPI + Ollama Vision API

This is a high-performance Python API that allows users to describe images using local multimodal LLMs (like Llama 3.2-Vision) via **Ollama**. It supports both binary file uploads (BLOBs) and local file paths.

## 📋 Features
* **Dual-Input Support:** Upload a file directly or provide a server-side file path.
* **Ollama Integration:** Leverages local GPU/CPU power for private image analysis.
* **Automatic Docs:** Interactive Swagger UI for testing without writing code.

---

## 🛠️ Prerequisites

1. **Ollama Installed:** [Download Ollama](https://ollama.com/) and ensure it is running.
2. **Vision Model:** Pull a multimodal model (Llama 3.2-Vision is recommended for speed and accuracy).
   ```bash
   ollama pull qwen3.5
   ```
3. **Python 3.9+**

---

## 🚀 Getting Started

### 1. Installation
Clone this repository or copy the `main.py` file, then install the dependencies:

```bash
pip install fastapi uvicorn ollama python-multipart
```

### 2. Run the Server
Start the FastAPI application using Uvicorn:

```bash
python -m uvicorn app:app --reload
```
*The server will start at `http://127.0.0.1:8000`.*

---

## 📡 API Usage & Examples

### Option 1: Uploading an Image (BLOB)
Use this when you want to send an image from your client machine to the API.

**CURL Command:**
```bash
curl -X POST http://127.0.0.1:8000/describe \
     -F "file=@/path/to/your/image.jpg"
```

### Option 2: Providing a File Path
Use this if the image already exists on the server's local storage.

**CURL Command:**
```bash
curl -X POST http://127.0.0.1:8000/describe \
     -F "path=/absolute/path/to/server/image.png"
```

---

## 🧪 Interactive Testing
FastAPI comes with a built-in UI for testing. 

1. Navigate to `http://127.0.0.1:8000/docs`.
2. Click on the **POST `/describe`** endpoint.
3. Click **"Try it out"**.
4. Use the **file** picker to upload an image OR type a path into the **path** field.
5. Hit **Execute**.

---

## ⚙️ Configuration
You can modify the model used by editing the `MODEL` variable in `main.py`:

| Model | Size | Requirement |
| :--- | :--- | :--- |
| `qwen3.5` | ~7B | Recommended (Balanced) |
| `llava` | ~7B | High Compatibility |
| `moondream` | ~1.6B | Fast / Low Resource |

---

## 📝 Expected Response
```json
{
  "description": "A close-up photo of a ginger tabby cat sitting on a wooden porch during sunset."
}
```


curl -X POST http://127.0.0.1:8001/classify \
     -F "file=@/path/to/your/image.jpg"