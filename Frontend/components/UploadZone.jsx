// components/UploadZone.jsx
import { useRef, useState } from "react";

const ACCEPTED = ["image/jpeg", "image/png", "image/webp", "image/bmp"];

export default function UploadZone({ onFileSelect }) {
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef(null);

  function handleFiles(fileList) {
    const file = fileList[0];
    if (!file) return;
    if (!ACCEPTED.includes(file.type)) {
      alert("Please upload a JPG, PNG, WEBP, or BMP image.");
      return;
    }
    onFileSelect(file);
  }

  function handleDrop(e) {
    e.preventDefault();
    setDragOver(false);
    handleFiles(e.dataTransfer.files);
  }

  return (
    <div className="upload-section">
      <label
        className={`upload-zone ${dragOver ? "drag-over" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp,image/bmp"
          onChange={(e) => handleFiles(e.target.files)}
        />
        <div className="upload-icon">
          <i className="ti ti-photo-up" aria-hidden="true"></i>
        </div>
        <div className="upload-label">Drop an animal photo here</div>
        <div className="upload-sub">or click to browse your files</div>
        <div className="upload-hint">Supports JPG, PNG, WEBP, BMP &middot; up to 10MB</div>
      </label>
    </div>
  );
}
