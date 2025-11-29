import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { Upload, Download, RefreshCw, Image as ImageIcon, Sparkles, Sun, Moon } from 'lucide-react';

function App() {
  const [originalImage, setOriginalImage] = useState(null);
  const [colorizedImage, setColorizedImage] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);
  const [theme, setTheme] = useState('dark');
  const fileInputRef = useRef(null);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prev => prev === 'dark' ? 'light' : 'dark');
  };

  const handleFileSelect = async (file) => {
    if (!file) return;

    // Preview original
    const reader = new FileReader();
    reader.onload = (e) => setOriginalImage(e.target.result);
    reader.readAsDataURL(file);

    setIsLoading(true);
    setColorizedImage(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:8000/api/colorize', formData, {
        responseType: 'blob',
      });

      const url = URL.createObjectURL(response.data);
      setColorizedImage(url);
    } catch (error) {
      console.error("Error colorizing image:", error);
      alert("Failed to colorize image. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const onDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const onDragOver = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const onDragLeave = (e) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDownload = () => {
    if (colorizedImage) {
      const link = document.createElement('a');
      link.href = colorizedImage;
      link.download = 'colorized_image.png';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  const reset = () => {
    setOriginalImage(null);
    setColorizedImage(null);
  };

  return (
    <div className="container">
      <button className="theme-toggle" onClick={toggleTheme} aria-label="Toggle theme">
        {theme === 'dark' ? <Sun size={24} /> : <Moon size={24} />}
      </button>

      <h1>SCA-NET ImageColorizer</h1>
      <p className="subtitle">Transform your black & white memories into vibrant reality with our self-calibrated AI model.</p>

      <div className="card">
        {!originalImage ? (
          <div
            className={`upload-zone ${isDragOver ? 'drag-active' : ''}`}
            onDrop={onDrop}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onClick={() => fileInputRef.current.click()}
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={(e) => handleFileSelect(e.target.files[0])}
              style={{ display: 'none' }}
              accept="image/*"
            />
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem' }}>
              <div style={{ background: 'rgba(99, 102, 241, 0.2)', padding: '1.5rem', borderRadius: '50%' }}>
                <Upload size={48} color="#818cf8" />
              </div>
              <h3 style={{ margin: 0, fontSize: '1.5rem', color: 'var(--text-main)' }}>Upload an Image</h3>
              <p style={{ color: 'var(--text-muted)', margin: 0 }}>Drag & drop or click to browse</p>
            </div>
          </div>
        ) : (
          <div>
            <div className="preview-container">
              <div className="image-box">
                <img src={originalImage} alt="Original" />
                <div className="image-label">Original Grayscale</div>
              </div>

              <div className="image-box">
                {isLoading ? (
                  <div style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', background: 'var(--image-box-bg)' }}>
                    <div className="loading-spinner"></div>
                    <p style={{ color: 'var(--text-muted)' }}>Colorizing...</p>
                  </div>
                ) : (
                  <>
                    <img src={colorizedImage} alt="Colorized" />
                    <div className="image-label" style={{ color: '#818cf8' }}>
                      <Sparkles size={16} style={{ display: 'inline', verticalAlign: 'text-bottom', marginRight: '4px' }} />
                      AI Colorized
                    </div>
                  </>
                )}
              </div>
            </div>

            <div style={{ display: 'flex', justifyContent: 'center', gap: '1rem', marginTop: '2rem' }}>
              <button className="btn btn-secondary" onClick={reset}>
                <RefreshCw size={20} />
                New Image
              </button>
              {colorizedImage && (
                <button className="btn" onClick={handleDownload}>
                  <Download size={20} />
                  Download Result
                </button>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
