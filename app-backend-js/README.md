This `README.md` is designed to be plug-and-play for your `app-backend-js` directory. It covers the environment setup, the server execution, and the specific `curl` commands to hit your LocalTunnel URL.

---

```markdown
# Smart City Image Description API (Ollama Bridge)

This is a Node.js-based bridge that exposes a local Ollama instance to the internet via LocalTunnel. It allows remote clients to upload images or provide local paths to get AI-generated descriptions using the `qwen3.5` model.

## 🚀 Setup Instructions

### 1. Prerequisites
* **Node.js**: v18 or higher installed.
* **Ollama**: Installed and running locally.
* **Model**: Ensure you have the model downloaded:
  ```bash
  ollama pull qwen3.5
  ```

### 2. Installation
Navigate to your project folder and install the dependencies:
```bash
npm install
```

### 3. Running the Server
Start the TypeScript execution using `ts-node`:
```bash
npx ts-node index.ts
```
The server will start on `http://localhost:3000`.

### 4. Exposing Globally (LocalTunnel)
In a **separate terminal window**, run LocalTunnel to generate your public URL:
```bash
npx localtunnel --port 3000
```
*Note: Copy the "your url is: ..." link provided in the terminal.*

---

## 🛠 Usage (Client API)

You can interact with the API using `curl`. Replace `YOUR_TUNNEL_URL` with the URL provided by LocalTunnel.

### Option A: Sending a Local File (Upload)
Use this when the image is located on the client's machine.
```bash
curl -X POST https://YOUR_TUNNEL_URL/describe \
  -H "bypass-tunnel-reminder: true" \
  -F "file=@/home/user/pictures/city_traffic.jpg"
```

### Option B: Providing a Server Path
Use this if the image is already on the server's filesystem.
```bash
curl -X POST https://YOUR_TUNNEL_URL/describe \
  -H "bypass-tunnel-reminder: true" \
  -F "path=/absolute/path/on/server/image.jpg"
```

---

## 📝 API Reference

| Field | Type | Description |
| :--- | :--- | :--- |
| `file` | `Buffer` | (Optional) The image file sent via multipart form-data. |
| `path` | `String` | (Optional) The absolute filesystem path to the image. |

**Success Response:**
```json
{
  "description": "A busy city intersection with several autonomous vehicles..."
}
```

## ⚠️ Troubleshooting
* **Bypass Warning**: LocalTunnel sometimes shows a splash screen. If your `curl` returns HTML instead of JSON, ensure you are sending the `bypass-tunnel-reminder: true` header.
* **Timeout**: Image processing can take time depending on your GPU. If the request times out, check the terminal running `index.ts` for Ollama progress.
```

---

### One final tip for your "Smart City" project:
If you find that LocalTunnel is occasionally flaky (as free tunnels sometimes are), an alternative you might look into later is **Cloudflare Tunnel (cloudflared)**. It's more stable for long-term "Smart City" management projects, but for quick testing and cracking code, your current LocalTunnel setup is perfect.