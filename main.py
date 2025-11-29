import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model import SCANet
from utils import ViCOWColorizationDataset, mse_metric, psnr_metric, ssim_metric

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for i, (L, AB) in enumerate(dataloader):
        L, AB = L.to(device), AB.to(device)
        optimizer.zero_grad()
        output = model(L)
        loss = criterion(output, AB)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 10 == 0:
            print(f"Batch {i}, Loss: {loss.item():.6f}")
    return running_loss / len(dataloader)

def validate_epoch(model, dataloader, device):
    model.eval()
    mse_total, psnr_total, ssim_total = 0, 0, 0
    with torch.no_grad():
        for L, AB in dataloader:
            L, AB = L.to(device), AB.to(device)
            output = model(L)
            mse_total += mse_metric(output, AB)
            psnr_total += psnr_metric(output, AB)
            ssim_total += ssim_metric(output, AB)
    count = len(dataloader)
    print(f"Validation MSE: {mse_total/count:.6f}, PSNR: {psnr_total/count:.2f}, SSIM: {ssim_total/count:.4f}")

def main():
    dataset_path = 'dataset'
    train_csv = os.path.join(dataset_path, 'train.csv')
    val_csv = os.path.join(dataset_path, 'val.csv')
    batch_size = 16
    epochs = 50
    lr = 1e-4
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor()
    ])

    train_dataset = ViCOWColorizationDataset(train_csv, dataset_path, transform=transform)
    val_dataset = ViCOWColorizationDataset(val_csv, dataset_path, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SCANet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    os.makedirs('checkpoints', exist_ok=True)
    for epoch in range(epochs):
        print(f"Epoch {epoch+1} Training...")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1} Train Loss: {train_loss:.6f}")

        print(f"Epoch {epoch+1} Validation...")
        validate_epoch(model, val_loader, device)

        ckpt_path = os.path.join('checkpoints', f'scanet_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

if __name__ == "__main__":
    main()

