// components/ImagePreview.jsx
import { useState, useEffect } from "react";

export default function ImagePreview({
  file,
  previewUrl,
  topK,
  setTopK,
  onPredict,
  onClear,
  loading,
  onFileSelect,
  hideImage = false,
}) {
  if (!file) return null;

  const sizeKB = (file.size / 1024).toFixed(1);
  const sizeDisplay = file.size > 1024 * 1024
    ? `${(file.size / (1024 * 1024)).toFixed(2)} MB`
    : `${sizeKB} KB`;

  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const img = new Image();
    img.src = previewUrl;
    img.onload = () => {
      setDimensions({ width: img.naturalWidth, height: img.naturalHeight });
    };
  }, [previewUrl]);

  return (
    <div className={`preview-card ${hideImage ? "preview-card--compact" : ""}`}>
      <div className="preview-header">
        <h3>Selected Image</h3>
        <div className="preview-actions">
          <label className="btn-predict" style={{ cursor: "pointer" }}>
            <input 
              type="file" 
              accept="image/*" 
              onChange={(e) => {
                if (e.target.files && e.target.files.length > 0) {
                  onFileSelect(e.target.files[0]);
                }
              }} 
              style={{ display: "none" }} 
            />
            <i className="ti ti-refresh" aria-hidden="true" style={{ fontSize: 14, verticalAlign: -2, marginRight: 4 }}></i>
            Predict Another Animal
          </label>
          <button className="btn-remove" onClick={onClear}>
            <i className="ti ti-x" aria-hidden="true" style={{ fontSize: 14, verticalAlign: -2, marginRight: 4 }}></i>
            Remove
          </button>
        </div>
      </div>

      <div className="preview-body">
        {!hideImage && (
          <div className="preview-img-wrap">
            <img src={previewUrl} alt="Selected animal" />
          </div>
        )}

        <div className="preview-info">
          <div className="file-meta">
            <div className="meta-row">
              <span>File name</span>
              <span title={file.name}>
                {file.name.length > 22 ? file.name.slice(0, 19) + "..." : file.name}
              </span>
            </div>
            <div className="meta-row">
              <span>File size</span>
              <span>{sizeDisplay}</span>
            </div>
            <div className="meta-row">
              <span>Format</span>
              <span>{file.type.split("/")[1]?.toUpperCase()}</span>
            </div>
            <div className="meta-row">
              <span>Dimensions</span>
              <span>{dimensions.width}×{dimensions.height}</span>
            </div>
          </div>


          <button className="predict-btn" onClick={onPredict} disabled={loading}>
            {loading ? (
              <>
                <i className="ti ti-loader-2" aria-hidden="true"></i>
                Analyzing...
              </>
            ) : (
              <>
                <i className="ti ti-sparkles" aria-hidden="true"></i>
                Identify animal
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
