import formidable from "formidable";
import FormData from "form-data";
import fs from "fs";
import fetch from "node-fetch";

export const config = {
  api: {
    bodyParser: false,
    maxDuration: 60,
  },
};

const BACKEND_URL =
  process.env.BACKEND_URL ||
  process.env.HF_BACKEND_URL ||
  process.env.FLASK_URL ||
  "https://sk-saniya-animal-vision-backend.hf.space";

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  const form = formidable({ keepExtensions: true });
  const [fields, files] = await form.parse(req);

  const file = Array.isArray(files.image) ? files.image[0] : files.image;
  const top_k = Array.isArray(fields.top_k) ? fields.top_k[0] : fields.top_k || "5";

  if (!file) {
    return res.status(400).json({ error: "No image file provided" });
  }

  const formData = new FormData();
  formData.append("image", fs.createReadStream(file.filepath), {
    filename: file.originalFilename || "upload.jpg",
    contentType: file.mimetype || "image/jpeg",
  });
  formData.append("top_k", top_k);

  try {
    const response = await fetch(`${BACKEND_URL}/predict`, {
      method: "POST",
      body: formData,
      headers: formData.getHeaders(),
    });

    const responseText = await response.text();
    let data;
    try {
      data = JSON.parse(responseText);
    } catch {
      data = { error: responseText || "Prediction server returned an invalid response." };
    }

    return res.status(response.status).json(data);
  } catch (err) {
    console.error("Prediction proxy error:", err);
    return res.status(502).json({
      error: "Could not reach prediction server. Make sure the backend is running and BACKEND_URL is correct.",
    });
  }
}
