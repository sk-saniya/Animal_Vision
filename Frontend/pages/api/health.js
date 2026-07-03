// pages/api/health.js
// Simple proxy to check if the Flask backend is online

const FLASK_URL = process.env.FLASK_URL || "http://localhost:7860";

export default async function handler(req, res) {
  try {
    const response = await fetch(`${FLASK_URL}/health`, { method: "GET" });
    const data = await response.json();
    return res.status(200).json({ online: true, ...data });
  } catch {
    return res.status(200).json({ online: false });
  }
}
