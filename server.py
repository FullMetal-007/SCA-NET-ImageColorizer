import os
import io
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms
from PIL import Image
from model import SCANet
from utils import lab_to_rgb

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SCANet().to(device)
CHECKPOINT_PATH = "checkpoints/scanet_epoch_50.pth"

if os.path.exists(CHECKPOINT_PATH):
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    print(f"Model loaded from {CHECKPOINT_PATH}")
else:
    print(f"Warning: Checkpoint not found at {CHECKPOINT_PATH}")

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

@app.post("/api/colorize")
async def colorize(file: UploadFile = File(...)):
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('L')
        
        # Preprocess
        L = transform(image).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            output = model(L)
        
        # Postprocess
        colorized_img = lab_to_rgb(L[0], output[0])
        
        # Save to buffer
        img_byte_arr = io.BytesIO()
        colorized_img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        return Response(content=img_byte_arr, media_type="image/png")
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
