import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler
from tqdm import tqdm
import random
import numpy as np
from torch.amp import autocast
import warnings
from PIL import Image, ImageFile

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore")

CONFIG = {
    'seed': 3407,
    'img_size': 512,
    'batch_size': 8,
    'accum_steps': 4,
    'ssl_epochs': 50,
    'lr': 5e-4,
    'device': torch.device('cuda'),
    'num_workers': 8
}


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(CONFIG['seed'])


class GPU_FrequencyGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('srm_kernel', torch.tensor(
            [[[0, 0, 0], [0, -1, 0], [0, 0, 0]], [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
             [[0, 0, 0], [0, -1, 0], [0, 0, 0]]], dtype=torch.float32).unsqueeze(1))

    def forward(self, x):
        rgb = (x - 0.5) / 0.5;
        x_f = x.float()
        fft = torch.fft.rfft2(x_f, norm='backward')
        fft_amp = torch.log1p(torch.abs(fft));
        fft_amp = (fft_amp - 3.0) / 3.0
        if fft_amp.shape[-1] != x.shape[-1]: fft_amp = torch.nn.functional.interpolate(fft_amp,
                                                                                       size=(x.shape[2], x.shape[3]),
                                                                                       mode='bilinear',
                                                                                       align_corners=False)
        srm = torch.nn.functional.conv2d(x_f, self.srm_kernel, padding=1, groups=3) * 3.0
        return torch.cat([rgb, fft_amp, srm], dim=1)


class LayerNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-6): super().__init__(); self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x): return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class GRN(nn.Module):
    def __init__(self, dim, eps=1e-6): super().__init__(); self.gamma = nn.Parameter(
        torch.zeros(1, 1, 1, dim)); self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim)); self.eps = eps

    def forward(self, x): Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True); Nx = Gx / (
                Gx.mean(dim=-1, keepdim=True) + self.eps); return self.gamma * (x * Nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    def __init__(self, dim): super().__init__(); self.dwconv = nn.Conv2d(dim, dim, 7, padding=3,
                                                                         groups=dim); self.norm = nn.LayerNorm(dim,
                                                                                                               eps=1e-6); self.pwconv1 = nn.Linear(
        dim, 4 * dim); self.act = nn.GELU(); self.grn = GRN(4 * dim); self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x): input = x; x = self.dwconv(x); x = x.permute(0, 2, 3, 1); x = self.norm(x); x = self.pwconv1(
        x); x = self.act(x); x = self.grn(x); x = self.pwconv2(x); return input + x.permute(0, 3, 1, 2)


class TitanNet(nn.Module):
    def __init__(self, in_chans=9, dims=[96, 192, 384, 768], depths=[3, 3, 12, 3]):
        super().__init__()
        self.preproc = GPU_FrequencyGenerator()
        self.downsamples = nn.ModuleList([nn.Sequential(nn.Conv2d(in_chans, dims[0], 4, 4), LayerNorm2d(dims[0]))])
        for i in range(3): self.downsamples.append(
            nn.Sequential(LayerNorm2d(dims[i]), nn.Conv2d(dims[i], dims[i + 1], 2, 2)))
        self.stages = nn.ModuleList()
        for i in range(4): self.stages.append(nn.Sequential(*[ConvNeXtV2Block(dims[i]) for _ in range(depths[i])]))
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)

    def forward(self, x):
        x = self.preproc(x)
        for i in range(4): x = self.downsamples[i](x); x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))


class SimMIM(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = nn.Sequential(nn.Conv2d(768, 9 * (32 ** 2), 1), nn.PixelShuffle(32))

    def forward(self, x, mask_ratio):
        with torch.no_grad(): x_9ch = self.encoder.module.preproc(x) if hasattr(self.encoder,
                                                                                'module') else self.encoder.preproc(x)
        B, C, H, W = x_9ch.shape
        mask = (torch.rand(B, 1, H // 32, W // 32, device=x.device) < mask_ratio).float()
        mask = torch.nn.functional.interpolate(mask, size=(H, W), mode='nearest')
        feat = x_9ch * (1 - mask)
        for i in range(4): feat = self.encoder.downsamples[i](feat); feat = self.encoder.stages[i](feat)
        x_rec = self.decoder(feat)
        return (torch.abs(x_9ch - x_rec) * mask).sum() / (mask.sum() + 1e-5)


class DiskAllImages(Dataset):
    def __init__(self, roots):
        self.files = []
        for root in roots:
            if os.path.exists(root):
                self.files.extend([os.path.join(root, f) for f in os.listdir(root) if f.endswith('.jpg')])
        print(f"Found {len(self.files)} images for SSL.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        try:
            img = cv2.imread(self.files[i]);
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img.shape[0] != 512: img = cv2.resize(img, (512, 512))
            return torch.from_numpy(np.transpose(img, (2, 0, 1)))
        except:
            return torch.zeros((3, 512, 512))


if __name__ == '__main__':
    ds = DiskAllImages(['data/faces_512/train', 'data/faces_512/test'])
    loader = DataLoader(ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=8, drop_last=True)

    simmim = SimMIM(TitanNet().to(CONFIG['device'])).to(CONFIG['device'])
    opt = optim.AdamW(simmim.parameters(), lr=CONFIG['lr'], weight_decay=0.05)

    scaler = GradScaler()

    print("STARTING SSL 512 (ALIGNED)...")
    for ep in range(CONFIG['ssl_epochs']):
        simmim.train()
        loss_acc = 0
        mask_ratio = 0.4
        opt.zero_grad()

        pbar = tqdm(loader, desc=f"SSL Ep {ep + 1}")
        for i, x in enumerate(pbar):
            x = x.to(CONFIG['device'], non_blocking=True).float().div(255.0)

            with autocast(device_type="cuda"):
                if random.random() < 0.5: x = torch.flip(x, [3])
                loss = simmim(x, mask_ratio) / CONFIG['accum_steps']

            scaler.scale(loss).backward()

            if (i + 1) % CONFIG['accum_steps'] == 0:
                scaler.step(opt);
                scaler.update();
                opt.zero_grad()

            loss_acc += loss.item() * CONFIG['accum_steps']
            pbar.set_postfix({'loss': loss.item() * CONFIG['accum_steps']})

        torch.save(simmim.encoder.state_dict(), 'titan_512_aligned_ssl.pth')
        print(f"SSL Ep {ep + 1} DONE.")