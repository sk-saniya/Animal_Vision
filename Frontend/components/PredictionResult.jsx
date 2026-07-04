// components/PredictionResult.jsx
import { useEffect, useState } from "react";

export default function PredictionResult({ result }) {
  const [animated, setAnimated] = useState(false);

  useEffect(() => {
    const t = setTimeout(() => setAnimated(true), 80);
    return () => clearTimeout(t);
  }, [result]);

  if (!result) return null;

  const predictedAnimal = result.predicted_animal || result.predictedAnimal || result.name;
  const accuracy = typeof result.accuracy === "number" ? result.accuracy : null;
  const predictions = Array.isArray(result.predictions) ? result.predictions : [];
  const top = predictions[0];
  const maxConf = predictions[0]?.confidence || 1;
  const safeFileName = result.file_name || result.filename || null;

  return (
    <div className="results-card">
      {/* Header */}
      <div className="results-header">
        <h2>Prediction Results</h2>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          {predictions.length > 0 && (
            <span className="results-meta">{predictions.length} predictions</span>
          )}
        </div>
      </div>

      {/* Top prediction banner */}
      {predictedAnimal && (
        <div className="prediction-box">
          <div className="prediction-box-label">Predicted Animal</div>
          <div className="prediction-box-name">{predictedAnimal}</div>
          {safeFileName && (
            <div className="prediction-box-accuracy" style={{ marginTop: "0.4rem" }}>
              File: <strong>{safeFileName}</strong>
            </div>
          )}
          <div className="prediction-box-accuracy">
            Accuracy: <strong>{accuracy !== null ? `${accuracy.toFixed(2)}%` : "—"}</strong>
          </div>
        </div>
      )}

      {top && (
        <div className="top-result">
          <span className="top-result-rank">#1 Match</span>
          <span className="top-result-name">{top.label}</span>
          <span className="top-result-conf">{top.confidence.toFixed(2)}%</span>
        </div>
      )}

      {/* Rest of predictions */}
      {predictions.length > 0 && (
        <div className="predictions-list">
          {predictions.map((pred, i) => {
            const barClass =
              i === 0 ? "rank-1" : i === 1 ? "rank-2" : i === 2 ? "rank-3" : "rank-other";
            const pct = animated ? (pred.confidence / maxConf) * 100 : 0;
            return (
              <div className="prediction-row" key={i}>
                <span className="pred-rank">{i + 1}</span>
                <span className="pred-name" title={pred.label}>{pred.label}</span>
                <div className="pred-bar-wrap">
                  <div
                    className={`pred-bar ${barClass}`}
                    style={{ width: `${pct}%` }}
                  />
                </div>
                <span className="pred-conf">{pred.confidence.toFixed(4)}%</span>
              </div>
            );
          })}
        </div>
      )}

      {/* Not an animal result — shown inside the same results card */}
      {result.notAnimal && (
        <div className="prediction-box" style={{ textAlign: "center" }}>
          <div style={{ fontSize: "2.5rem", marginBottom: "0.5rem" }}>🚫🐾</div>
          <div className="prediction-box-label">Not an Animal</div>
          <div
            className="prediction-box-name"
            style={{ fontSize: "1.1rem", color: "var(--danger, #e74c3c)" }}
          >
            {result.message || "This image does not appear to contain an animal."}
          </div>
          <div className="prediction-box-accuracy" style={{ marginTop: "0.5rem" }}>
            Please upload a photo of an animal to get a prediction.
          </div>
        </div>
      )}

      {!predictedAnimal && !result.notAnimal && predictions.length === 0 && (
        <div className="prediction-box">
          <div className="prediction-box-label">Prediction</div>
          <div className="prediction-box-name">No prediction available</div>
          <div className="prediction-box-accuracy">Try another image.</div>
        </div>
      )}
    </div>
  );
}
