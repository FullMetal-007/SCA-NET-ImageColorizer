# SCA-NET ImageColorizer

This project is an advanced deep learning application for **automatic image colorization**, featuring a **Self-Calibrated AutoEncoder (SCA-Net)** architecture and a modern **React-based Web UI**.

It combines a powerful PyTorch backend for accurate color restoration with a user-friendly interface for easy interaction.

---

## Features

### Core Model
**Self-Calibrated Convolutional AutoEncoder**: Enhanced texture consistency and color balance.
**High-Performance Inference**: Optimized for 256x256 resolution.
**Metrics**: PSNR, SSIM, and MSE evaluation support.
**Dataset Support**: Compatible with COCO, CelebA-HQ, Places365.

### Web Interface
**Modern UI**: Sleek React application with Light/Dark mode support.
**Drag & Drop**: Intuitive image upload.
**Instant Preview**: Side-by-side comparison of grayscale and colorized results.
**Download**: One-click export of colorized images.
**FastAPI Backend**: Robust and fast model serving.

---

## Project Structure

```text
DL-PBL-copy/
├── client/              # React Frontend
│   ├── src/
│   └── ...
├── checkpoints/         # Saved models (scanet_epoch_50.pth)
├── dataset/             # Training data
├── outputs/             # Inference outputs
├── inference.py         # Single-image inference script
├── main.py              # Training pipeline
├── model.py             # SCA-Net architecture
├── server.py            # FastAPI backend server
├── utils.py             # Utilities & metrics
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## Setup & Installation

### 1. Prerequisites
- Python 3.8+
- Node.js & npm
- CUDA-capable GPU (recommended)

### 2. Backend Setup
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### 3. Frontend Setup
Navigate to the client directory and install dependencies:
```bash
cd client
npm install
```

---

## Usage

### Option A: Run the Web Application (Recommended)

1.  **Start the Backend**:
    ```bash
    # From the root directory
    python server.py
    ```
    The API will start at `http://localhost:8000`.

2.  **Start the Frontend**:
    ```bash
    # From the client directory
    cd client
    npm run dev
    ```
    The UI will open at `http://localhost:5173`.

### Option B: Train the Model
To train the model from scratch:
```bash
python main.py --epochs 50 --batch-size 16
```
- MSE loss is printed per batch.
- PSNR and SSIM are computed after each epoch.
- Checkpoints are saved to `checkpoints/`.

### Option C: CLI Inference
To colorize a single image via command line:
```bash
python inference.py
```
*(Note: Edit `inference.py` to specify input/output paths)*

---

## Model Overview: SCA-Net

The **Self-Calibrated AutoEncoder (SCA-Net)** improves upon standard U-Nets by incorporating self-calibrated convolutions. This allows the model to:
- **Recalibrate Features**: Adaptively build long-range spatial and inter-channel dependencies.
- **Enhance Details**: Better preservation of texture and edges.
- **Stabilize Colors**: More natural and consistent color prediction.

**Architecture Flow:**
`Input (L) → Encoder (SC Blocks) → Bottleneck → Decoder (Upsampling) → Output (AB)`

---

## Acknowledgements

- **lukemelas/Automatic-Image-Colorization** for the original framework inspiration.
- **SCNet (Liu et al., CVPR 2020)** for the Self-Calibrated Convolution concepts.
