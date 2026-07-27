// components/PredictionResult.jsx

export default function PredictionResult({ result, imageMode }) {
  if (!result) return null;

  const predictedAnimal = result.predicted_animal || result.predictedAnimal || result.name;
  const accuracy = typeof result.accuracy === "number" ? result.accuracy : null;
  const detections = Array.isArray(result.detections) ? result.detections : [];
  const detectedAnimalsImage = result.annotated_image || result.annotatedImage;
  const isSingleAnimalInMultipleMode =
    result.singleAnimalImage || (imageMode === "multiple" && detections.length === 1);

  return (
    <div className="results-card">
      <div className="results-header">
        <h2>Prediction Results</h2>
      </div>

      {detectedAnimalsImage && !isSingleAnimalInMultipleMode && (
        <div className="results-image-wrap">
          <img
            className="results-image"
            src={detectedAnimalsImage}
            alt="Detected animals with bounding boxes"
          />
        </div>
      )}

      {isSingleAnimalInMultipleMode && (
        <div className="result-notice">
          <i className="ti ti-info-circle" aria-hidden="true"></i>
          <div>
            <div className="result-notice-title">Single animal image</div>
            <div className="result-notice-text">
              This image contains only one animal.
            </div>
            <div className="result-notice-action">
              Please use Single Animal Image mode for this upload.
            </div>
          </div>
        </div>
      )}

      {result.multipleAnimals && (
        <div className="result-notice">
          <i className="ti ti-info-circle" aria-hidden="true"></i>
          <div>
            <div className="result-notice-title">Multiple animals detected</div>
            <div className="result-notice-text">
              This image contains multiple animals.
            </div>
            <div className="result-notice-action">
              Please use the Multiple Animal Image section.
            </div>
          </div>
        </div>
      )}

      {detections.length > 0 && !isSingleAnimalInMultipleMode && (
        <div className="detections-list">
          {detections.map((detection) => (
            <div className="detection-row" key={detection.label || detection.name}>
              {detection.crop_image && (
                <img
                  className="detection-thumb"
                  src={detection.crop_image}
                  alt={`${detection.name || "Detected animal"} crop`}
                />
              )}
              <div className="detection-info">
                <div className="detection-label">Object #{detection.label}</div>
                <div className="detection-name">{detection.name || "Detected animal"}</div>
              </div>
              <div className="detection-accuracy">
                {typeof detection.accuracy === "number"
                  ? `${detection.accuracy.toFixed(2)}%`
                  : "-"}
              </div>
            </div>
          ))}
        </div>
      )}

      {predictedAnimal && !isSingleAnimalInMultipleMode && (
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
