import { useState, useEffect, useCallback } from "react";
import Head from "next/head";
import UploadZone from "../components/UploadZone";
import ImagePreview from "../components/ImagePreview";
import PredictionResult from "../components/PredictionResult";

const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL ||
  "https://sk-saniya-animal-vision-backend.hf.space";

export default function Home() {
  const backgroundAnimals = [
    { name: "Bee", icon: "🐝", className: "bg-animal bg-animal-1" },
    { name: "Cat", icon: "🐱", className: "bg-animal bg-animal-2" },
    { name: "Deer", icon: "🦌", className: "bg-animal bg-animal-3" },
    { name: "Dog", icon: "🐶", className: "bg-animal bg-animal-4" },
    { name: "Dolphin", icon: "🐬", className: "bg-animal bg-animal-5" },
    { name: "Duck", icon: "🦆", className: "bg-animal bg-animal-6" },
    { name: "Elephant", icon: "🐘", className: "bg-animal bg-animal-7" },
    { name: "Flamingo", icon: "🦩", className: "bg-animal bg-animal-8" },
    { name: "Flying fish", icon: "🐟", className: "bg-animal bg-animal-9" },
    { name: "Giraffe", icon: "🦒", className: "bg-animal bg-animal-10" },
    { name: "Turtle", icon: "🐢", className: "bg-animal bg-animal-11" },
    { name: "Tiger", icon: "🐯", className: "bg-animal bg-animal-12" },
    { name: "Swan", icon: "🦢", className: "bg-animal bg-animal-13" },
    { name: "Starfish", icon: "⭐", className: "bg-animal bg-animal-14" },
    { name: "Snake", icon: "🐍", className: "bg-animal bg-animal-15" },
    { name: "Rabbit", icon: "🐰", className: "bg-animal bg-animal-16" },
    { name: "Penguin", icon: "🐧", className: "bg-animal bg-animal-17" },
    { name: "Peacock", icon: "🦚", className: "bg-animal bg-animal-18" },
    { name: "Parrot", icon: "🦜", className: "bg-animal bg-animal-19" },
    { name: "Panda", icon: "🐼", className: "bg-animal bg-animal-20" },
    { name: "Owl", icon: "🦉", className: "bg-animal bg-animal-21" },
    { name: "Fox", icon: "🦊", className: "bg-animal bg-animal-22" },
    { name: "Cockroach", icon: "🪳", className: "bg-animal bg-animal-23" },
    { name: "Butterfly", icon: "🦋", className: "bg-animal bg-animal-24" },
    { name: "Lion", icon: "🦁", className: "bg-animal bg-animal-25" },
    { name: "Cow", icon: "🐄", className: "bg-animal bg-animal-26" },
    { name: "Crow", icon: "🐦", className: "bg-animal bg-animal-27" },
    { name: "Camel", icon: "🐫", className: "bg-animal bg-animal-28" },
    { name: "Crab", icon: "🦀", className: "bg-animal bg-animal-29" },
    { name: "Zebra", icon: "🦓", className: "bg-animal bg-animal-30" },
    { name: "Lama", icon: "🦙", className: "bg-animal bg-animal-31" },
  ];

  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [topK, setTopK] = useState(5);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [errorDetail, setErrorDetail] = useState(null);
  const [backendOnline, setBackendOnline] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);

  useEffect(() => {
    console.log("[Animal Vision] Checking backend health:", BACKEND_URL);
    fetch(`${BACKEND_URL}/health`)
      .then((res) => res.json())
      .then((data) => {
        console.log("[Animal Vision] Backend health OK:", data);
        setBackendOnline(true);
        if (data) {
          setModelInfo({
            labels: data.animals_loaded,
            device: data.device,
          });
        }
      })
      .catch(() => setBackendOnline(false));
  }, []);

  const handleFileSelect = useCallback((selectedFile) => {
    setFile(selectedFile);
    setResult(null);
    setError(null);
    setErrorDetail(null);

    const reader = new FileReader();
    reader.onload = () => {
      setPreviewUrl(reader.result);
    };
    reader.readAsDataURL(selectedFile);
  }, []);

  function handleClear() {
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    setErrorDetail(null);
  }

  async function handlePredict() {
    if (!file) return;
    setLoading(true);
    setError(null);
    setErrorDetail(null);
    setResult(null);
    console.log("[Animal Vision] Sending prediction request:", {
      backendUrl: BACKEND_URL,
      fileName: file.name,
      fileSize: file.size,
      fileType: file.type,
      topK,
    });

    const formData = new FormData();
    formData.append("image", file);
    formData.append("top_k", topK);

    try {
      const res = await fetch(`${BACKEND_URL}/predict`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json().catch(() => ({}));
      console.log("[Animal Vision] Predict response:", {
        status: res.status,
        ok: res.ok,
        data,
      });

      if (!res.ok) {
        const message = data?.error || data?.detail || "Prediction failed";
        const detailParts = [
          `HTTP ${res.status}`,
          data?.detail ? `detail: ${data.detail}` : null,
          data?.error ? `error: ${data.error}` : null,
          data?.animal_score !== undefined ? `animal_score: ${data.animal_score}` : null,
        ].filter(Boolean);

        throw {
          response: {
            status: res.status,
            data,
          },
          message,
          detail: detailParts.join(" | "),
        };
      }

      setResult(data);
    } catch (err) {
      console.error("[Animal Vision] Predict request failed:", err);
      const responseData = err.response?.data;
      const msg = err.message || responseData?.error || "Something went wrong while predicting. Please try again.";

      if (msg.toLowerCase().includes("not an animal")) {
        setResult({
          notAnimal: true,
          message: responseData?.detail || msg,
          animalScore: responseData?.animal_score,
        });
      } else {
        setError(msg);
        setErrorDetail(err.detail || responseData?.detail || null);
      }
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <Head>
        <title>Animal Vision &middot; CLIP Classifier</title>
        <meta name="description" content="Identify any of 25,000 animal species from a photo using CLIP." />
        <link rel="icon" type="image/svg+xml" href="/brain-svgrepo-com.svg" />
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/tabler-icons/2.44.0/iconfont/tabler-icons.min.css" />
      </Head>

      <div className="background-animals" aria-hidden="true">
        {backgroundAnimals.map((animal) => (
          <span key={animal.name} className={animal.className}>
            <span className="bg-animal-icon">{animal.icon}</span>
          </span>
        ))}
      </div>

      <header className="header">
        <div className="container header-inner">
          <div className="header-logo">
            <img src="/brain-svgrepo-com.svg" alt="" aria-hidden="true" className="header-logo-img" />
          </div>
          <span className="header-title">Animal Vision</span>
          <span className="header-sub">
            {backendOnline === null ? (
              "Checking server..."
            ) : null}
          </span>
        </div>
      </header>

      <main className="container" style={{ flex: 1, paddingBottom: "3rem" }}>
        <div className="hero">
          <h2 className="hero-subtitle">ANIMAL VISION</h2>
          <h1 className="hero-title">Identify Any Animal From A Single Photo</h1>
          <p className="hero-description">
            Upload an image and get fast AI-powered animal identification with live predictions,
            top matches, and a simple, easy-to-use interface.
          </p>
        </div>

        {!file ? (
          <UploadZone onFileSelect={handleFileSelect} />
        ) : (
          <div className="side-by-side">
            <div className="side-left">
              <ImagePreview
                file={file}
                previewUrl={previewUrl}
                topK={topK}
                setTopK={setTopK}
                onPredict={handlePredict}
                onClear={handleClear}
                loading={loading}
                onFileSelect={handleFileSelect}
              />
            </div>

            <div className="side-right">
              {loading && (
                <div className="loading-card">
                  <div className="spinner"></div>
                </div>
              )}

              {error && !loading && (
                <div className="error-card">
                  <i className="ti ti-alert-triangle error-icon" aria-hidden="true" style={{ color: "var(--danger)" }}></i>
                  <div>
                    <div className="error-title">Prediction failed</div>
                    <div className="error-msg">{error}</div>
                    {errorDetail && <div className="error-detail">{errorDetail}</div>}
                  </div>
                </div>
              )}

              {result && !loading && (
                <PredictionResult result={result} />
              )}
            </div>
          </div>
        )}

        {!file && !result && (
          <div className="stats-row">
            <div className="stat-card">
              <div className="stat-value">25,000</div>
              <div className="stat-label">Animal species</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">Zero-shot</div>
              <div className="stat-label">No training needed</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">&lt;2s</div>
              <div className="stat-label">Avg. response time</div>
            </div>
          </div>
        )}
      </main>

      <footer className="footer">
        <div className="container footer-inner">
          <p>Made with <span className="heart">❤</span> for animal vision. © {new Date().getFullYear()} Animal Vision. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}
