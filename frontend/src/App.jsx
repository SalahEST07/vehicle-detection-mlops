
import { useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

function App() {
  const [mode, setMode] = useState("image");

  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);

  const [video, setVideo] = useState(null);
  const [videoPreview, setVideoPreview] = useState(null);
  const [videoResults, setVideoResults] = useState(null);

  const [confidence, setConfidence] = useState(0.25);
  const [results, setResults] = useState(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleImageChange = (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
      setError("Please select a JPEG, PNG, or WEBP image.");
      return;
    }

    if (preview) URL.revokeObjectURL(preview);

    setImage(file);
    setPreview(URL.createObjectURL(file));
    setResults(null);
    setError("");
  };

  const handleVideoChange = (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const allowedTypes = [
      "video/mp4",
      "video/avi",
      "video/x-msvideo",
      "video/quicktime",
    ];

    if (
      !allowedTypes.includes(file.type) ||
      ![".mp4", ".avi", ".mov"].includes(
        file.name.toLowerCase().slice(file.name.lastIndexOf("."))
      )
    ) {
      setError("Please select an MP4, AVI, or MOV video.");
      return;
    }

    if (videoPreview) URL.revokeObjectURL(videoPreview);

    setVideo(file);
    setVideoPreview(URL.createObjectURL(file));
    setVideoResults(null);
    setError("");
  };

  const handleDetection = async () => {
    if (!image) {
      setError("Please upload an image first.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const formData = new FormData();
      formData.append("file", image);

      const response = await fetch(
        `${API_URL}/predict?confidence=${confidence}`,
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Prediction failed.");
      }

      setResults(data);
    } catch (err) {
      setError(err.message || "Could not connect to the API.");
    } finally {
      setLoading(false);
    }
  };

  const handleVideoDetection = async () => {
    if (!video) {
      setError("Please upload a video first.");
      return;
    }

    setLoading(true);
    setError("");
    setVideoResults(null);

    try {
      const formData = new FormData();
      formData.append("file", video);

      const response = await fetch(
        `${API_URL}/predict/video?confidence=${confidence}`,
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Video processing failed.");
      }

      setVideoResults({
        ...data,
        video_url: `${API_URL}/videos/${encodeURIComponent(data.output_filename)}`,
      });
    } catch (err) {
      setError(err.message || "Could not connect to the API.");
    } finally {
      setLoading(false);
    }
  };

  const switchMode = (newMode) => {
    setMode(newMode);
    setError("");
    setResults(null);
    setVideoResults(null);
  };

  const vehicleClasses = [
    { name: "Car", icon: "🚗" },
    { name: "Motorcycle", icon: "🏍️" },
    { name: "Bus", icon: "🚌" },
    { name: "Truck", icon: "🚚" },
  ];

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <div className="brand-icon">V</div>
          <div>
            <h1>VehicleVision</h1>
            <p>AI Vehicle Detection System</p>
          </div>
        </div>

        <div className="api-status">
          <span className="status-dot"></span>
          Detection Dashboard
        </div>
      </header>

      <main className="main-content">
        <section className="hero">
          <span className="eyebrow">COMPUTER VISION • YOLO26n</span>
          <h2>Intelligent Vehicle Detection</h2>
          <p>
            Upload an image or video to detect cars, motorcycles, buses,
            and trucks using your trained deep learning model.
          </p>
        </section>

        <div className="mode-selector">
          <button
            className={`mode-button ${mode === "image" ? "active" : ""}`}
            onClick={() => switchMode("image")}
          >
            🖼️ Image Detection
          </button>

          <button
            className={`mode-button ${mode === "video" ? "active" : ""}`}
            onClick={() => switchMode("video")}
          >
            🎬 Video Detection
          </button>
        </div>

        <div className="dashboard">
          <section className="panel upload-panel">
            <div className="panel-heading">
              <div>
                <h3>{mode === "image" ? "Image Input" : "Video Input"}</h3>
                <p>
                  {mode === "image"
                    ? "Choose an image to analyze"
                    : "Choose a video to process"}
                </p>
              </div>
              <span className="step-number">01</span>
            </div>

            {mode === "image" ? (
              <label className="upload-area">
                <input
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  onChange={handleImageChange}
                />

                {preview ? (
                  <img
                    src={preview}
                    alt="Selected preview"
                    className="preview"
                  />
                ) : (
                  <div className="upload-placeholder">
                    <div className="upload-icon">↑</div>
                    <strong>Click to upload an image</strong>
                    <span>JPEG, PNG or WEBP</span>
                  </div>
                )}
              </label>
            ) : (
              <label className="upload-area video-upload-area">
                <input
                  type="file"
                  accept="video/mp4,video/avi,video/x-msvideo,video/quicktime,.mp4,.avi,.mov"
                  onChange={handleVideoChange}
                />

                {videoPreview ? (
                  <video
                    src={videoPreview}
                    className="preview"
                    controls
                  />
                ) : (
                  <div className="upload-placeholder">
                    <div className="upload-icon">▶</div>
                    <strong>Click to upload a video</strong>
                    <span>MP4, AVI or MOV</span>
                  </div>
                )}
              </label>
            )}

            {(image || video) && (
              <p className="selected-file">
                Selected: {mode === "image" ? image?.name : video?.name}
              </p>
            )}

            <div className="confidence-control">
              <div className="control-label">
                <label htmlFor="confidence">Confidence Threshold</label>
                <strong>{Math.round(confidence * 100)}%</strong>
              </div>

              <input
                id="confidence"
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={confidence}
                onChange={(event) =>
                  setConfidence(Number(event.target.value))
                }
              />

              <div className="range-labels">
                <span>0%</span>
                <span>100%</span>
              </div>
            </div>

            <button
              className="detect-button"
              onClick={
                mode === "image" ? handleDetection : handleVideoDetection
              }
              disabled={
                loading || (mode === "image" ? !image : !video)
              }
            >
              {loading
                ? mode === "image"
                  ? "Analyzing image..."
                  : "Processing video..."
                : mode === "image"
                  ? "Run Vehicle Detection"
                  : "Process Video"}
            </button>

            {error && <p className="error-message">{error}</p>}
          </section>

          <section className="panel results-panel">
            <div className="panel-heading">
              <div>
                <h3>Detection Results</h3>
                <p>Analysis output</p>
              </div>
              <span className="step-number">02</span>
            </div>

            {mode === "image" ? (
              !results ? (
                <div className="empty-results">
                  <div className="result-icon">◎</div>
                  <strong>Waiting for analysis</strong>
                  <p>Upload an image and run detection to see results here.</p>
                </div>
              ) : (
                <div className="results-content">
                  <div className="total-detections">
                    <span>Total detected vehicles</span>
                    <strong>{results.total_detections}</strong>
                  </div>

                  {results.detections.length === 0 ? (
                    <p>No vehicles detected at this confidence threshold.</p>
                  ) : (
                    results.detections.map((detection, index) => (
                      <div className="detection-item" key={index}>
                        <div>
                          <strong>{detection.class_name}</strong>
                          <span>Class ID: {detection.class_id}</span>
                        </div>
                        <strong>
                          {(detection.confidence * 100).toFixed(2)}%
                        </strong>
                      </div>
                    ))
                  )}
                </div>
              )
            ) : !videoResults ? (
              <div className="empty-results">
                <div className="result-icon">▶</div>
                <strong>Waiting for video analysis</strong>
                <p>Upload a video and process it to see results here.</p>
              </div>
            ) : (
              <div className="results-content">
                <div className="total-detections">
                  <span>Processed frames</span>
                  <strong>{videoResults.processed_frames}</strong>
                </div>

                
<div className="video-statistics">
  <h4>Unique Vehicles Detected</h4>

  {vehicleClasses.map((vehicle) => {
    const count =
      videoResults.detection_counts[
        vehicle.name.toLowerCase()
      ] ?? 0;

    return (
      <div className="detection-item" key={vehicle.name}>
        <div>
          <strong>
            {vehicle.icon} {vehicle.name}
          </strong>
        </div>
        <strong>{count.toLocaleString()}</strong>
      </div>
    );
  })}

  <div className="detection-item total-vehicles">
    <strong>Total Unique Vehicles</strong>
    <strong>
      {Object.values(videoResults.detection_counts)
        .reduce((total, count) => total + count, 0)
        .toLocaleString()}
    </strong>
  </div>
</div>


                <div className="video-output">
                  <h4>Annotated Video</h4>

                  <video
                    className="annotated-video"
                    src={videoResults.video_url}
                    controls
                    playsInline
                  />

                  <a
                    className="download-button"
                    href={videoResults.video_url}
                    download={videoResults.output_filename}
                  >
                    ↓ Download Annotated Video
                  </a>
                </div>
              </div>
            )}
          </section>
        </div>

        <section className="classes-section">
          <h3>Supported Vehicle Classes</h3>
          <div className="class-list">
            {vehicleClasses.map((vehicle) => (
              <div className="class-card" key={vehicle.name}>
                <span>{vehicle.icon}</span>
                <strong>{vehicle.name}</strong>
              </div>
            ))}
          </div>
        </section>

        <footer>
          <p>VehicleVision · Powered by YOLO26n and ONNX Runtime</p>
        </footer>
      </main>
    </div>
  );
}

export default App;
