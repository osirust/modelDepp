import os
import cv2
import math
import glob
import random
import shutil
import inspect
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler
from contextlib import contextmanager
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from tqdm import tqdm
from PIL import Image, ImageFile
import warnings
import torchvision.transforms.functional as TF
import datetime

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
cv2.setNumThreads(0)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore")

CONFIG = {
    'seed': 3407,
    'img_size': 384,
    'batch_size': 8,
    'accum_steps': 4,
    'ssl_epochs': 150,
    'clf_epochs': 25,
    'lr': 3e-4,
    'n_folds': 5,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'num_workers': 8,
    'use_amp': True
}


def log(message):
    print(message, flush=True)
    with open("training_v42_log.txt", "a", encoding="utf-8") as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {message}\n")


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


seed_everything(CONFIG['seed'])
log(f"TITAN-X V42 DIAMOND | 384px | SSL 150ep | Device: {CONFIG['device']}")


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
        if fft_amp.shape[-1] != x.shape[-1]: fft_amp = F.interpolate(fft_amp, size=(x.shape[2], x.shape[3]),
                                                                     mode='bilinear', align_corners=False)
        srm = F.conv2d(x_f, self.srm_kernel, padding=1, groups=3) * 3.0
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
        x_9ch = self.encoder.module.preproc(x) if hasattr(self.encoder, 'module') else self.encoder.preproc(x)
        B, C, H, W = x_9ch.shape
        mask = (torch.rand(B, 1, H // 32, W // 32, device=x.device) < mask_ratio).float()
        mask = F.interpolate(mask, size=(H, W), mode='nearest')
        feat = x_9ch * (1 - mask)
        for i in range(4): feat = self.encoder.downsamples[i](feat); feat = self.encoder.stages[i](feat)
        x_rec = self.decoder(feat)
        return (torch.abs(x_9ch - x_rec) * mask).sum() / (mask.sum() + 1e-5)


class SimpleHead(nn.Module):
    def __init__(self, in_features=768):
        super().__init__()
        self.head = nn.Sequential(nn.Dropout(0.25), nn.Linear(in_features, 1))

    def forward(self, x): return self.head(x)


def smooth_labels(y, eps=0.1):
    return y * (1 - eps) + 0.5 * eps


class DiskTitanDataset(Dataset):
    def __init__(self, df, root_train, root_test, labeled=True):
        self.df = df
        self.labeled = labeled
        self.root_train = root_train
        self.root_test = root_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        fid = str(row.Id)

        path = os.path.join(self.root_train, f"{fid}.jpg")
        if not os.path.exists(path):
            path = os.path.join(self.root_test, f"{fid}.jpg")

        try:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_t = torch.from_numpy(np.transpose(img, (2, 0, 1)))
        except:
            img_t = torch.zeros((3, CONFIG['img_size'], CONFIG['img_size']))

        if self.labeled:
            return img_t, torch.tensor(float(row.target_feature), dtype=torch.float32)
        return img_t


class DiskAllImages(Dataset):
    def __init__(self, roots):
        self.files = []
        for root in roots:
            if os.path.exists(root):
                self.files.extend([os.path.join(root, f) for f in os.listdir(root) if f.endswith('.jpg')])
        log(f"SSL Dataset: Found {len(self.files)} images.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        try:
            img = cv2.imread(self.files[i]);
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return torch.from_numpy(np.transpose(img, (2, 0, 1)))
        except:
            return torch.zeros((3, CONFIG['img_size'], CONFIG['img_size']))


def run_ssl_pretrain():
    if os.path.exists('titan_v42_ssl.pth'):
        log("Found titan_v42_ssl.pth, skipping SSL...")
        return

    log("\nSTEP 1: SSL PRETRAIN (150 EPOCHS)...")
    ds = DiskAllImages(['data/faces_512/train', 'data/faces_512/test'])
    loader = DataLoader(ds, batch_size=CONFIG['batch_size'], shuffle=True,
                        num_workers=8, drop_last=True, pin_memory=True)

    backbone = TitanNet().to(CONFIG['device'])
    simmim = SimMIM(backbone).to(CONFIG['device'])
    opt = optim.AdamW(simmim.parameters(), lr=5e-4, weight_decay=0.05)

    scaler = GradScaler(enabled=CONFIG['use_amp'])

    for ep in range(CONFIG['ssl_epochs']):
        simmim.train()
        loss_acc = 0
        mask_ratio = 0.4
        opt.zero_grad()

        pbar = tqdm(loader, desc=f"SSL Ep {ep + 1}", leave=False)
        for i, x in enumerate(pbar):
            x = x.to(CONFIG['device'], non_blocking=True).float().div(255.0)

            from torch.amp import autocast
            with autocast(device_type="cuda", enabled=CONFIG['use_amp']):
                if random.random() < 0.5: x = torch.flip(x, [3])
                loss = simmim(x, mask_ratio) / CONFIG['accum_steps']

            scaler.scale(loss).backward()

            if (i + 1) % CONFIG['accum_steps'] == 0:
                scaler.step(opt);
                scaler.update();
                opt.zero_grad()

            loss_acc += loss.item() * CONFIG['accum_steps']
            pbar.set_postfix({'loss': loss.item() * CONFIG['accum_steps']})

        log(f"SSL Ep {ep + 1} | Loss: {loss_acc / len(loader):.4f}")

        if (ep + 1) % 10 == 0:
            torch.save(simmim.encoder.state_dict(), f'titan_v42_ssl_ep{ep + 1}.pth')

    torch.save(simmim.encoder.state_dict(), 'titan_v42_ssl.pth')
    log("SSL DONE.")


def run_classification():
    log("\nSTEP 2: CLASSIFICATION (5 FOLDS)...")

    # Check for Diamond CSV (Pseudo Labels)
    CSV_PATH = 'train_diamond.csv'
    if not os.path.exists(CSV_PATH):
        log(" Diamond CSV not found, using standard solution.")
        CSV_PATH = 'data/dataset/train_solution.csv'

    df = pd.read_csv(CSV_PATH)
    if 'id' in df.columns: df.rename(columns={'id': 'Id'}, inplace=True)
    if 'target' in df.columns: df.rename(columns={'target': 'target_feature'}, inplace=True)

    full_ds = DiskTitanDataset(df, 'data/faces_512/train', 'data/faces_512/test')
    kf = StratifiedKFold(n_splits=CONFIG['n_folds'], shuffle=True, random_state=CONFIG['seed'])

    fold_thresholds = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.zeros(len(df)), df.target_feature)):
        log(f"\n FOLD {fold + 1}/{CONFIG['n_folds']}")

        if os.path.exists(f'v42_fold{fold}.pth'):
            log(f"Fold {fold} done. Skipping...")
            continue

        train_loader = DataLoader(torch.utils.data.Subset(full_ds, train_idx),
                                  batch_size=CONFIG['batch_size'], shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(torch.utils.data.Subset(full_ds, val_idx),
                                batch_size=CONFIG['batch_size'], shuffle=False, num_workers=8, pin_memory=True)

        backbone = TitanNet().to(CONFIG['device'])
        head = SimpleHead().to(CONFIG['device'])

        if os.path.exists('titan_v42_ssl.pth'):
            log("Loading SSL Weights...")
            st = torch.load('titan_v42_ssl.pth', map_location=CONFIG['device'], weights_only=False)
            backbone.load_state_dict({k.replace('module.', ''): v for k, v in st.items()}, strict=False)

        opt = optim.AdamW(list(backbone.parameters()) + list(head.parameters()), lr=CONFIG['lr'], weight_decay=0.05)
        scaler = GradScaler(enabled=CONFIG['use_amp'])
        criterion = nn.BCEWithLogitsLoss()

        best_f1 = 0

        for ep in range(CONFIG['clf_epochs']):
            backbone.train();
            head.train()
            loss_acc = 0
            opt.zero_grad()
            pbar = tqdm(train_loader, desc=f"Ep {ep + 1}", leave=False)

            for i, (x, y) in enumerate(pbar):
                x = x.to(CONFIG['device'], non_blocking=True).float().div(255.0)
                y = y.to(CONFIG['device'], non_blocking=True)

                from torch.amp import autocast
                with autocast(device_type="cuda", enabled=CONFIG['use_amp']):
                    logits = head(backbone(x)).view(-1)
                    loss = criterion(logits, smooth_labels(y)) / CONFIG['accum_steps']

                scaler.scale(loss).backward()

                if (i + 1) % CONFIG['accum_steps'] == 0:
                    scaler.step(opt);
                    scaler.update();
                    opt.zero_grad()

                loss_acc += loss.item() * CONFIG['accum_steps']
                pbar.set_postfix({'loss': loss.item() * CONFIG['accum_steps']})

            backbone.eval();
            head.eval()
            probs, targs = [], []
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(CONFIG['device'], non_blocking=True).float().div(255.0)
                    with autocast(device_type="cuda", enabled=CONFIG['use_amp']):
                        p = torch.sigmoid(head(backbone(x))).view(-1)
                    probs.extend(p.cpu().numpy())
                    targs.extend(y.cpu().numpy())

            curr_f1, best_th = 0, 0.5
            for th in np.linspace(0.1, 0.9, 50):
                f = f1_score(targs, (np.array(probs) > th).astype(int))
                if f > curr_f1: curr_f1 = f; best_th = th

            log(f"Ep {ep + 1} | Val F1: {curr_f1:.4f} (th={best_th:.2f})")

            if curr_f1 > best_f1:
                best_f1 = curr_f1
                torch.save(backbone.state_dict(), f'v42_backbone_fold{fold}.pth')
                torch.save(head.state_dict(), f'v42_head_fold{fold}.pth')
                torch.save({'th': best_th}, f'v42_meta_fold{fold}.pth')

    log("Classification Training Done.")


def run_inference():
    log("\n STEP 3: TTA INFERENCE & SUBMISSION...")

    test_files = [f for f in os.listdir('data/faces_512/test') if f.endswith('.jpg')]
    test_ids = sorted([int(f.split('.')[0]) for f in test_files])

    class InferDS(Dataset):
        def __init__(self, ids): self.ids = ids

        def __len__(self): return len(self.ids)

        def __getitem__(self, i):
            path = os.path.join('data/faces_512/test', f"{self.ids[i]}.jpg")
            img = cv2.imread(path);
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return torch.from_numpy(np.transpose(img, (2, 0, 1)))

    loader = DataLoader(InferDS(test_ids), batch_size=CONFIG['batch_size'], shuffle=False, num_workers=8)

    final_logits = np.zeros(len(test_ids))
    valid_folds = 0

    for fold in range(CONFIG['n_folds']):
        if not os.path.exists(f'v42_backbone_fold{fold}.pth'): continue
        valid_folds += 1

        backbone = TitanNet().to(CONFIG['device'])
        backbone.load_state_dict(torch.load(f'v42_backbone_fold{fold}.pth', weights_only=False))
        head = SimpleHead().to(CONFIG['device'])
        head.load_state_dict(torch.load(f'v42_head_fold{fold}.pth', weights_only=False))

        backbone.eval();
        head.eval()
        fold_logits = []

        with torch.no_grad():
            for x in tqdm(loader, desc=f"Infer Fold {fold}"):
                x = x.to(CONFIG['device']).float().div(255.0)

                from torch.amp import autocast
                with autocast(device_type="cuda", enabled=CONFIG['use_amp']):
                    l1 = head(backbone(x)).view(-1)
                    l2 = head(backbone(torch.flip(x, [3]))).view(-1)
                    l3 = head(backbone(TF.rotate(x, 5))).view(-1)
                    l4 = head(backbone(TF.rotate(x, -5))).view(-1)

                    avg_logit = (l1 + l2 + l3 + l4) / 4.0
                    fold_logits.extend(avg_logit.cpu().numpy())

        final_logits += np.array(fold_logits)

    if valid_folds > 0:
        final_logits /= valid_folds
        final_probs = 1 / (1 + np.exp(-final_logits))  # Sigmoid

        ths = []
        for i in range(5):
            if os.path.exists(f'v42_meta_fold{i}.pth'):
                ths.append(torch.load(f'v42_meta_fold{i}.pth', weights_only=False)['th'])
        avg_th = np.mean(ths) if ths else 0.5

        sub = pd.DataFrame({'Id': test_ids, 'target_feature': (final_probs > avg_th).astype(int)})
        sub.to_csv('submission_v42.csv', index=False)
        log(f"Submission Saved! Avg Threshold: {avg_th:.4f}")


if __name__ == '__main__':
    run_ssl_pretrain()
    run_classification()

    run_inference()
