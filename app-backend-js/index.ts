import express, { Request, Response } from 'express';
import multer from 'multer';
import ollama from 'ollama';
import fs from 'fs-extra';

const app = express();
const upload = multer({ storage: multer.memoryStorage() });
const PORT = 3000;
const MODEL = "qwen3.5";

app.use(express.json());

app.post('/describe', upload.single('file'), async (req: any, res: any) => {
    console.log("📩 Received request at /describe");
    try {
        const file = req.file;
        const localPath = req.body.path;
        let imageSource: Buffer;

        if (file) {
            console.log("🖼️ Processing uploaded file...");
            imageSource = file.buffer;
        } else if (localPath) {
            console.log(`📂 Reading file from path: ${localPath}`);
            if (!(await fs.pathExists(localPath))) {
                return res.status(404).json({ detail: "Local path not found" });
            }
            imageSource = await fs.readFile(localPath);
        } else {
            return res.status(400).json({ detail: "No file or path provided" });
        }

        console.log("🤖 Sending to Ollama...");
        const response = await ollama.generate({
            model: MODEL,
            prompt: "Describe this image.",
            images: [imageSource]
        });

        console.log("✅ Success!");
        res.json({ description: response.response });

    } catch (error: any) {
        console.error("❌ Error:", error.message);
        res.status(500).json({ detail: error.message });
    }
});

app.listen(PORT, () => {
    console.log(`🚀 Server is screaming for attention at http://localhost:${PORT}`);
    console.log(`🛠️ Using model: ${MODEL}`);
});