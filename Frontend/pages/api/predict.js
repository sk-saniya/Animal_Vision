import formidable from "formidable";
import fs from "fs/promises";
import os from "os";

export const config = {
  api: {
    bodyParser: false,
  },
};

const BACKEND_URL =
  process.env.BACKEND_URL || process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:7860";

function parseForm(req) {
  const form = formidable({
    multiples: false,
    keepExtensions: true,
    uploadDir: os.tmpdir(),
  });

  return new Promise((resolve, reject) => {
    form.parse(req, (err, fields, files) => {
      if (err) return reject(err);
      resolve({ fields, files });
    });
  });
}

async function readUploadedFile(file) {
  if (!file) return null;

  const filePath = file.filepath || file.path;
  const originalName = file.originalFilename || file.name || "upload";
  const data = await fs.readFile(filePath);

  return { data, originalName, mimeType: file.mimetype || "application/octet-stream" };
}

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return res.status(405).json({ error: "Method not allowed" });
  }

  try {
    const { files } = await parseForm(req);
    const uploaded = files.image;
    const file = Array.isArray(uploaded) ? uploaded[0] : uploaded;

    if (!file) {
      return res.status(400).json({ error: "No image file provided" });
    }

    const forwardedFile = await readUploadedFile(file);
    const formData = new FormData();
    formData.append(
      "image",
      new Blob([forwardedFile.data], { type: forwardedFile.mimeType }),
      forwardedFile.originalName
    );

    const response = await fetch(`${BACKEND_URL.replace(/\/$/, "")}/predict`, {
      method: "POST",
      body: formData,
    });

    const payload = await response.json();
    return res.status(response.status).json(payload);
  } catch (error) {
    return res.status(500).json({
      error: "Prediction proxy failed",
      detail: error?.message || String(error),
    });
  }
}
