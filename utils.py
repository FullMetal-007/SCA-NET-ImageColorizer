import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from skimage import color
import numpy as np
import torch.nn.functional as F
from math import log10
from skimage.metrics import structural_similarity as ssim

class ViCOWColorizationDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        color_path = os.path.join(self.root_dir, row['colorPath'])
        gray_path = os.path.join(self.root_dir, row['grayPath'])

        color_img = Image.open(color_path).convert('RGB')
        gray_img = Image.open(gray_path).convert('L')

        if self.transform:
            color_img = self.transform(color_img)
            gray_img = self.transform(gray_img)
        else:
            color_img = transforms.ToTensor()(color_img)
            gray_img = transforms.ToTensor()(gray_img)

        color_np = color_img.permute(1,2,0).numpy()
        lab = color.rgb2lab(color_np).astype(np.float32)

        L = lab[:,:,0] / 100.0
        AB = lab[:,:,1:] / 128.0

        L = torch.tensor(L).unsqueeze(0)
        AB = torch.tensor(AB).permute(2,0,1)

        return L, AB

def mse_metric(output, target):
    return F.mse_loss(output, target).item()

def psnr_metric(output, target):
    mse = F.mse_loss(output, target)
    if mse == 0:
        return float('inf')
    return 10 * log10(1 / mse.item())

def ssim_metric(output, target):
    """Compute average SSIM over batch on CPU numpy arrays."""
    output_np = output.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    ssim_vals = []
    for i in range(output_np.shape[0]):
        out_img = np.transpose(output_np[i], (1,2,0))   # HWC
        tgt_img = np.transpose(target_np[i], (1,2,0))   # HWC
        # Rescale to original ranges
        out_img = (out_img * 128.0) + 128.0             # approximate AB channel scale
        tgt_img = (tgt_img * 128.0) + 128.0
        # Compute SSIM on each AB channel and average
        ssim_a = ssim(tgt_img[:,:,0], out_img[:,:,0], data_range=255)
        ssim_b = ssim(tgt_img[:,:,1], out_img[:,:,1], data_range=255)
        ssim_vals.append((ssim_a+ssim_b)/2)
    return np.mean(ssim_vals).item()

def lab_to_rgb(L, AB):
    L = L.cpu().numpy() * 100.0
    AB = AB.cpu().numpy() * 128.0
    lab = np.zeros((L.shape[1], L.shape[2], 3), dtype=np.float32)
    lab[:, :, 0] = L[0]
    lab[:, :, 1:] = AB.transpose(1,2,0)
    import cv2
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    rgb = np.clip(rgb, 0, 1)
    rgb = (rgb * 255).astype(np.uint8)
    from PIL import Image
    return Image.fromarray(rgb)
