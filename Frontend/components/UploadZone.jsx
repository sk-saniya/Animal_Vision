// components/UploadZone.jsx
import { useRef, useState } from "react";

const ACCEPTED = ["image/jpeg", "image/png", "image/webp", "image/bmp"];

export default function UploadZone({ onFileSelect, imageMode, setImageMode }) {
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
      <div className="upload-mode-actions" aria-label="Choose image upload mode">
        <button
          type="button"
          className={`upload-mode-btn ${imageMode === "single" ? "active" : ""}`}
          onClick={() => {
            setImageMode("single");
            inputRef.current?.click();
          }}
        >
          Single Animal Image
        </button>
        <button
          type="button"
          className={`upload-mode-btn ${imageMode === "multiple" ? "active" : ""}`}
          onClick={() => {
            setImageMode("multiple");
            inputRef.current?.click();
          }}
        >
          Multiple Animal Image
        </button>
      </div>
      <input
        ref={inputRef}
        type="file"
        style={{ display: "none" }}
        accept="image/jpeg,image/png,image/webp,image/bmp"
        onChange={(e) => handleFiles(e.target.files)}
      />
    </div>
  );
}
