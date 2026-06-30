// pages/api/predict.js
// Proxies multipart/form-data from Next.js to the Flask backend.
// Requires: npm install form-data node-fetch@2

import formidable from "formidable";
import FormData from "form-data";
import fs from "fs";
import fetch from "node-fetch";

export const config = {
  api: { bodyParser: false },   // must be off for file uploads
};

const FLASK_URL = process.env.FLASK_URL || "http://localhost:5000";

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  // Parse incoming multipart form
  const form = formidable({ keepExtensions: true });
  const [fields, files] = await form.parse(req);

  const file = Array.isArray(files.image) ? files.image[0] : files.image;
  const top_k = Array.isArray(fields.top_k) ? fields.top_k[0] : fields.top_k || "5";

  if (!file) {
    return res.status(400).json({ error: "No image file provided" });
  }

  // Forward to Flask
  const formData = new FormData();
  formData.append("image", fs.createReadStream(file.filepath), {
    filename: file.originalFilename || "upload.jpg",
    contentType: file.mimetype || "image/jpeg",
  });
  formData.append("top_k", top_k);

  try {
    const response = await fetch(`${FLASK_URL}/predict`, {
      method: "POST",
      body: formData,
      headers: formData.getHeaders(),
    });

    const data = await response.json();
    return res.status(response.status).json(data);
  } catch (err) {
    console.error("Flask proxy error:", err.message);
    return res.status(502).json({ error: "Could not reach prediction server. Make sure clip_api.py is running." });
  }
}
