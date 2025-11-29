import os
import torch
from torchvision import transforms
from PIL import Image
from model import SCANet
from utils import lab_to_rgb

def colorize_image(model_path, grayscale_path, save_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gray_img = Image.open(grayscale_path).convert('L')
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor()
    ])
    L = transform(gray_img).unsqueeze(0).to(device)
    model = SCANet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    with torch.no_grad():
        output = model(L)
    colorized = lab_to_rgb(L[0], output[0])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    colorized.save(save_path)
    print(f"Saved colorized image to {os.path.abspath(save_path)}")

if __name__ == "__main__":
    model_ckpt = "checkpoints/scanet_epoch_50.pth"
    gray_path = "C://Users//shourya//Downloads//PinDown.io_1755433388.jpg"
    output_path = "outputs/PinDown.io_1755433388_colorized.jpg"
    colorize_image(model_ckpt, gray_path, output_path)

#"C:\Users\shourya\Downloads\PinDown.io_1755433388.jpg"