// components/PredictionResult.jsx

export default function PredictionResult({ result }) {
  if (!result) return null;

  const predictedAnimal = result.predicted_animal || result.predictedAnimal || result.name;
  const accuracy = typeof result.accuracy === "number" ? result.accuracy : null;

  return (
    <div className="results-card">
      <div className="results-header">
        <h2>Prediction Results</h2>
      </div>

      {predictedAnimal && (
        <div className="prediction-hero">
          <div className="prediction-hero-label">Predicted Animal</div>
          <div className="prediction-hero-name">{predictedAnimal}</div>
          <div className="prediction-hero-accuracy">
            Accuracy: {accuracy !== null ? `${accuracy.toFixed(2)}%` : "—"}
          </div>
        </div>
      )}

      {result.notAnimal && (
        <div className="prediction-hero prediction-hero--warning">
          <div className="prediction-hero-label">Not an Animal</div>
          <div className="prediction-hero-name prediction-hero-name--warning">
            {result.message || "This image does not appear to contain an animal."}
          </div>
          <div className="prediction-hero-accuracy prediction-hero-accuracy--warning">
            Please upload a photo of an animal to get a prediction.
          </div>
        </div>
      )}
    </div>
  );
}
