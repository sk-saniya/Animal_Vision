

export const config = {
  api: {
    maxDuration: 10,
  },
};

const BACKEND_URL =
  process.env.BACKEND_URL ||
  process.env.HF_BACKEND_URL ||
  process.env.FLASK_URL ||
  "https://sk-saniya-animal-vision-backend.hf.space";

export default async function handler(req, res) {
  try {
    const response = await fetch(`${BACKEND_URL}/health`, { method: "GET" });
    const data = await response.json();
    return res.status(200).json({ online: true, ...data });
  } catch {
    return res.status(200).json({ online: false });
  }
}
