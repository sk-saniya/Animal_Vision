import { useMemo, useState } from "react";

const DEFAULT_API_URL = "http://localhost:7860";

export default function Home() {
  const [apiUrl, setApiUrl] = useState(
    process.env.NEXT_PUBLIC_API_URL || DEFAULT_API_URL
  );
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const canSubmit = useMemo(() => !!file && !loading, [file, loading]);

  const onFileChange = (event) => {
    const nextFile = event.target.files?.[0] || null;
    setFile(nextFile);
    setResult(null);
    setError("");

    if (preview) {
      URL.revokeObjectURL(preview);
    }

    setPreview(nextFile ? URL.createObjectURL(nextFile) : "");
  };

  const onSubmit = async (event) => {
    event.preventDefault();
    if (!file) return;

    setLoading(true);
    setResult(null);
    setError("");

    try {
      const formData = new FormData();
      formData.append("image", file);

      const response = await fetch(`${apiUrl.replace(/\/$/, "")}/predict`, {
        method: "POST",
        body: formData,
      });

      const payload = await response.json();

      if (!response.ok) {
        setError(payload.detail || payload.error || "Prediction failed");
        return;
      }

      setResult(payload);
    } catch (err) {
      setError(err?.message || "Unable to reach the backend");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="page-shell">
      <section className="hero">
        <div className="hero-copy">
          <p className="eyebrow">Animal Vision</p>
          <h1>Upload an image and identify the animal.</h1>
          <p className="lede">
            The frontend talks directly to the Flask backend and shows the
            prediction returned by CLIP.
          </p>
        </div>

        <form className="panel" onSubmit={onSubmit}>
          <label className="field">
            <span>Backend URL</span>
            <input
              value={apiUrl}
              onChange={(event) => setApiUrl(event.target.value)}
              placeholder="http://localhost:7860"
            />
          </label>

          <label className="upload">
            <input
              type="file"
              accept="image/*"
              onChange={onFileChange}
            />
            <span>{file ? file.name : "Choose an image to classify"}</span>
          </label>

          {preview ? (
            <div className="preview">
              <img src={preview} alt="Selected upload preview" />
            </div>
          ) : null}

          <button type="submit" disabled={!canSubmit}>
            {loading ? "Classifying..." : "Classify image"}
          </button>

          {result ? (
            <div className="result success">
              <strong>{result.predicted_animal}</strong>
              <span>Confidence: {result.accuracy}%</span>
            </div>
          ) : null}

          {error ? (
            <div className="result error">
              <strong>Unable to classify</strong>
              <span>{error}</span>
            </div>
          ) : null}
        </form>
      </section>
    </main>
  );
}
