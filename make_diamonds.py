import os
import cv2
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

CONFIG = {
    'img_size': 256,
    'batch_size': 128,
    'device': torch.device('cuda'),
    'conf_threshold': 0.995,
    'num_workers': 4
}

print(f"GENERATING DIAMONDS (Background Task)...")


class GPU_FrequencyGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('srm_kernel', torch.tensor(
            [[[0, 0, 0], [0, -1, 0], [0, 0, 0]], [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
             [[0, 0, 0], [0, -1, 0], [0, 0, 0]]], dtype=torch.float32).unsqueeze(1))

    def forward(self, x):
        rgb = (x - 0.5) / 0.5;
        x_f = x.float()
        fft = torch.fft.rfft2(x_f, norm='backward');
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
    def __init__(self, in_chans=9, dims=[96, 192, 384, 768], depths=[3, 3, 9, 3]):  # V38 structure
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


class ArcFaceHead(nn.Module):
    def __init__(self, in_features=768, out_features=2, s=30.0, m=0.50):
        super().__init__();
        self.s = s;
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features));
        nn.init.xavier_uniform_(self.weight)
        self.aux_head = nn.Linear(in_features, out_features)

    def forward(self, x, label=None, use_arcface=True):
        if not use_arcface: return self.aux_head(x)
        x = F.layer_norm(x, x.shape[1:]);
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        return cosine * self.s


class InferDS(Dataset):
    def __init__(self, ids, root):
        self.ids = ids; self.root = root

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        fid = str(self.ids[i])
        path = os.path.join(self.root, f"{fid}.jpg")
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((256, 256, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        return torch.from_numpy(np.transpose(img, (2, 0, 1))), int(self.ids[i])


if __name__ == '__main__':
    test_dir = 'data/dataset/test_images'
    if not os.path.exists(test_dir): test_dir = 'data/faces_test'
    test_files = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
    test_ids = sorted([int(f.split('.')[0]) for f in test_files])

    loader = DataLoader(InferDS(test_ids, test_dir), batch_size=CONFIG['batch_size'], shuffle=False,
                        num_workers=CONFIG['num_workers'])

    models = []
    print("Loading V38 weights...")
    for i in range(5):
        bk_path = f'titan_backbone_fold{i}.pth'
        hd_path = f'titan_head_fold{i}.pth'
        if os.path.exists(bk_path) and os.path.exists(hd_path):
            bk = TitanNet().to(CONFIG['device']).eval()
            hd = ArcFaceHead().to(CONFIG['device']).eval()
            st_bk = torch.load(bk_path, map_location=CONFIG['device'], weights_only=False)
            st_hd = torch.load(hd_path, map_location=CONFIG['device'], weights_only=False)
            if 'model' in st_bk: st_bk = st_bk['model']
            bk.load_state_dict({k.replace('module.', ''): v for k, v in st_bk.items()}, strict=False)
            hd.load_state_dict({k.replace('module.', ''): v for k, v in st_hd.items()}, strict=False)
            models.append((bk, hd))
            print(f"  Fold {i} loaded")

    if not models:
        print(" CRITICAL: No V38 models found! Cannot generate pseudo labels.")
        exit(1)

    print("Running Inference...")
    final_probs = []
    ids_list = []

    with torch.no_grad():
        for x, ids in tqdm(loader):
            x = x.to(CONFIG['device']).float().div(255.0)
            batch_probs = torch.zeros(x.shape[0], device=CONFIG['device'])
            from torch.cuda.amp import autocast

            with autocast():
                for bk, hd in models:
                    p1 = torch.softmax(hd(bk(x), None, use_arcface=True), dim=1)[:, 1]
                    p2 = torch.softmax(hd(bk(torch.flip(x, [3])), None, use_arcface=True), dim=1)[:, 1]
                    batch_probs += (p1 + p2) / 2
            batch_probs /= len(models)
            final_probs.extend(batch_probs.cpu().numpy())
            ids_list.extend(ids.numpy())

    pseudo_data = []
    for pid, prob in zip(ids_list, final_probs):
        if prob > CONFIG['conf_threshold']:
            pseudo_data.append({'Id': pid, 'target_feature': 1, 'is_pseudo': True})
        elif prob < (1 - CONFIG['conf_threshold']):
            pseudo_data.append({'Id': pid, 'target_feature': 0, 'is_pseudo': True})

    print(f"Diamonds found: {len(pseudo_data)}")

    orig_csv = 'data/dataset/train_solution.csv'
    if not os.path.exists(orig_csv): orig_csv = 'data/train_solution.csv'
    df_orig = pd.read_csv(orig_csv)
    if 'id' in df_orig.columns: df_orig.rename(columns={'id': 'Id'}, inplace=True)
    if 'target' in df_orig.columns: df_orig.rename(columns={'target': 'target_feature'}, inplace=True)
    df_orig['is_pseudo'] = False

    df_final = pd.concat([df_orig[['Id', 'target_feature', 'is_pseudo']], pd.DataFrame(pseudo_data)])
    df_final.to_csv('train_diamond.csv', index=False)
    print(f"Saved 'train_diamond.csv'. Main training will pick this up automatically.")