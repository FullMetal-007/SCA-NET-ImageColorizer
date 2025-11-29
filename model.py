import torch
import torch.nn as nn
import torch.nn.functional as F

class SCBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out

class SCANet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = SCBlock(1, 64)
        self.enc2 = SCBlock(64, 128)
        self.enc3 = SCBlock(128, 256)

        self.pool = nn.MaxPool2d(2,2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)     # upsample to [B,128,H/4,W/4]
        self.dec3 = SCBlock(384, 128)  # up3 (128) + e3 (256) = 384

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)      # upsample to [B,64,H/2,W/2]
        self.dec2 = SCBlock(192, 64)   # up2 (64) + e2 (128) = 192

        self.final = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)        # [B,64,H,W]
        p1 = self.pool(e1)       # [B,64,H/2,W/2]
        e2 = self.enc2(p1)       # [B,128,H/2,W/2]
        p2 = self.pool(e2)       # [B,128,H/4,W/4]
        e3 = self.enc3(p2)       # [B,256,H/4,W/4]
        p3 = self.pool(e3)       # [B,256,H/8,W/8]
        b = self.bottleneck(p3)  # [B,256,H/8,W/8]
        
        up3 = self.up3(b)        # [B,128,H/4,W/4]
        cat3 = torch.cat([up3, e3], dim=1)    # [B,384,H/4,W/4]
        d3 = self.dec3(cat3)     # [B,128,H/4,W/4]
        
        up2 = self.up2(d3)       # [B,64,H/2,W/2]
        cat2 = torch.cat([up2, e2], dim=1)    # [B,192,H/2,W/2]
        d2 = self.dec2(cat2)     # [B,64,H/2,W/2]
        
        out = self.final(d2)     # [B,2,H/2,W/2]
        out = F.interpolate(out, size=x.size()[2:], mode='bilinear', align_corners=False)
        return out
