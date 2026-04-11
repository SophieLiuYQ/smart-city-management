import express, { Request, Response } from 'express';
import multer from 'multer';
import ollama from 'ollama';
import fs from 'fs-extra';
import path from 'path';

const app = express();
const upload = multer({ storage: multer.memoryStorage() }); // Store files in memory as buffers
const PORT = 3000;
const MODEL = "qwen3.5";

// Middleware to parse form data (for the 'path' field)
app.use(express.urlencoded({ extended: true }));

app.post('/describe', upload.single('file'), async (req: Request, res: Response) => {
    try {
        const file = req.file;
        const localPath = req.body.path;
        let imageSource: string | Buffer;

        // 1. Handle File Upload (Buffer)
        if (file) {
            imageSource = file.buffer;
        } 
        // 2. Handle Local Path (Fallback)
        else if (localPath) {
            if (!(await fs.pathExists(localPath))) {
                return res.status(404).json({ detail: "Local path not found" });
            }
            // Read the file into a buffer to send to Ollama
            imageSource = await fs.readFile(localPath);
        } 
        else {
            return res.status(400).json({ detail: "Provide either a file or a path" });
        }

        // 3. Call Ollama
        const response = await ollama.generate({
            model: MODEL,
            prompt: "Describe this image.",
            images: [imageSource]
        });

        res.json({ description: response.response });

    } catch (error: any) {
        console.error(error);
        res.status(500).json({ detail: error.message });
    }
});

app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});