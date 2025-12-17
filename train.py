import os
import cv2
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from tqdm import tqdm
from PIL import Image, ImageFile
from facenet_pytorch import MTCNN
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
import subprocess

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore")

CONFIG = {
    'seed': 42,
    'img_size': 256,
    'batch_size': 128,
    'accum_steps': 1,
    'epochs': 20,
    'lr': 1e-3,
    'min_lr': 1e-6,
    'n_folds': 5,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'tta': True,
    'num_workers': 24,
    'compile_model': True
}


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


seed_everything(CONFIG['seed'])
GPU_COUNT = torch.cuda.device_count()
print(f"Device: {CONFIG['device']} | GPUs available: {GPU_COUNT}")
if GPU_COUNT > 1:
    print(f"Multi-GPU Activated: Using {GPU_COUNT} x RTX 5080")


def download_data():
    # Если папка уже есть, не качаем заново
    if os.path.exists('data/faces_train') and len(os.listdir('data/faces_train')) > 1000:
        print("Данные уже на месте, пропускаем скачивание.")
        return

    if not os.path.exists('data'):
        print(" Настройка Kaggle API...")
        if not os.path.exists('kaggle.json'):
            raise FileNotFoundError("ОШИБКА: Файл 'kaggle.json' не найден!")

        os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
        os.system('cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json')

        print("Скачивание данных...")
        zip_path = 'data/ml-intensive-yandex-academy-autumn-2025.zip'
        try:
            subprocess.run(
                ['kaggle', 'competitions', 'download', '-c', 'ml-intensive-yandex-academy-autumn-2025', '-p', 'data'],
                check=True)
            print(" Распаковка...")
            subprocess.run(['unzip', '-q', '-o', zip_path, '-d', 'data/'], check=True)

            print(" Удаление архива для экономии места...")
            if os.path.exists(zip_path):
                os.remove(zip_path)

            print(" Данные готовы, место освобождено.")
        except Exception as e:
            print(f"Ошибка: {e}")
            if os.path.exists(zip_path): os.remove(zip_path)
            exit(1)


download_data()

DIRS = {'train': 'data/faces_train', 'test': 'data/faces_test'}
for d in DIRS.values(): os.makedirs(d, exist_ok=True)

print("Инициализация MTCNN...")
mtcnn = MTCNN(keep_all=False, select_largest=True, margin=30, post_process=False, device='cuda:0')


def check_genai_traces(path):
    try:
        with open(path, 'rb') as f:
            head = f.read(8192)
            f.seek(0, 2);
            eof = f.tell();
            f.seek(max(0, eof - 8192));
            tail = f.read()
        data = head + tail
        signatures = [b'c2pa', b'C2PA', b'JUMBF', b'SynthID', b'AI Generated', b'adobe:ns:meta']
        if any(s in data for s in signatures): return 1.0

        img = Image.open(path)
        exif = img.getexif()
        if exif:
            for _, val in exif.items():
                if any(
                    x in str(val).lower() for x in ['midjourney', 'dall-e', 'stable diffusion', 'firefly']): return 1.0
    except:
        pass
    return 0.0


def smart_process(src_dir, dst_dir, ids):
    # Проверка, чтобы не делать двойную работу
    existing = set([int(f.split('.')[0]) for f in os.listdir(dst_dir)])
    to_process = [x for x in ids if x not in existing]

    if not to_process:
        print(f"Папка {dst_dir} готова.")
        return

    print(f"Обработка {len(to_process)} изображений...")

    for pid in tqdm(to_process, desc="Preprocessing"):
        p_jpg = os.path.join(src_dir, f"{pid}.jpg")
        p_png = os.path.join(src_dir, f"{pid}.png")
        path = p_jpg if os.path.exists(p_jpg) else (p_png if os.path.exists(p_png) else None)
        if not path: continue

        try:
            img = Image.open(path).convert('RGB')
            boxes, _ = mtcnn.detect(img)

            if boxes is not None:
                box = boxes[0]
                x1, y1, x2, y2 = [int(b) for b in box]
                w, h = x2 - x1, y2 - y1
                pad = int(0.3 * max(w, h))
                img_w, img_h = img.size
                x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                x2, y2 = min(img_w, x2 + pad), min(img_h, y2 + pad)
                face = img.crop((x1, y1, x2, y2))
            else:
                w, h = img.size;
                s = min(w, h)
                face = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))

            face = face.resize((CONFIG['img_size'], CONFIG['img_size']), Image.BICUBIC)
            face.save(os.path.join(dst_dir, f"{pid}.jpg"), quality=100)
        except:
            Image.new('RGB', (CONFIG['img_size'], CONFIG['img_size'])).save(os.path.join(dst_dir, f"{pid}.jpg"))


train_df = pd.read_csv('data/dataset/train_solution.csv')
train_df.columns = ['Id', 'target_feature']
smart_process('data/dataset/train_images', DIRS['train'], train_df['Id'].values)

test_files = [int(f.split('.')[0]) for f in os.listdir('data/dataset/test_images') if f.endswith(('.jpg', '.png'))]
smart_process('data/dataset/test_images', DIRS['test'], test_files)


class BayarConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=0):
        super(BayarConv2d, self).__init__()
        self.in_channels = in_channels;
        self.out_channels = out_channels
        self.kernel_size = kernel_size;
        self.stride = stride;
        self.padding = padding
        self.kernel = nn.Parameter(torch.rand(self.in_channels, self.out_channels, kernel_size, kernel_size),
                                   requires_grad=True)

    def forward(self, x):
        ctr_idx = self.kernel_size // 2
        real_kernel = self.kernel.clone()
        mask = torch.ones_like(real_kernel)
        mask[:, :, ctr_idx, ctr_idx] = 0
        real_kernel = real_kernel * mask
        kernel_sum = real_kernel.sum(dim=(2, 3), keepdim=True)
        real_kernel = real_kernel / (kernel_sum + 1e-7)
        real_kernel[:, :, ctr_idx, ctr_idx] = -1.0
        return F.conv2d(x, real_kernel, stride=self.stride, padding=self.padding)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip);
        self.act = nn.SiLU()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x;
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        return identity * torch.sigmoid(self.conv_w(x_w)) * torch.sigmoid(self.conv_h(x_h))


class ResNestBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c);
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c);
        self.ca = CoordAtt(out_c, out_c)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(nn.Conv2d(in_c, out_c, 1, stride, bias=False), nn.BatchNorm2d(out_c))

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.ca(self.bn2(self.conv2(out)))
        return self.act(out + self.shortcut(x))


class DualStreamNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_stem = nn.Sequential(nn.Conv2d(3, 64, 3, 2, 1, bias=False), nn.BatchNorm2d(64), nn.SiLU())
        self.rgb_layers = nn.Sequential(self._make_layer(64, 64, 2), self._make_layer(64, 128, 2),
                                        self._make_layer(128, 256, 2))
        self.bayar = BayarConv2d(3, 3, padding=2)
        self.noise_stem = nn.Sequential(nn.Conv2d(3, 64, 3, 2, 1, bias=False), nn.BatchNorm2d(64), nn.SiLU())
        self.noise_layers = nn.Sequential(self._make_layer(64, 64, 2), self._make_layer(64, 128, 2),
                                          self._make_layer(128, 256, 2))
        self.fusion = nn.Sequential(nn.Conv2d(512, 512, 1, bias=False), nn.BatchNorm2d(512), nn.SiLU())
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.4), nn.Linear(512, 1))

    def _make_layer(self, in_c, out_c, blocks):
        layers = [ResNestBlock(in_c, out_c, stride=2)]
        for _ in range(blocks - 1): layers.append(ResNestBlock(out_c, out_c, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        r = self.rgb_layers(self.rgb_stem(x))
        n = self.noise_layers(self.noise_stem(self.bayar(x)))
        return self.head(self.fusion(torch.cat([r, n], dim=1)))


class ComboLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        tp = (probs * targets).sum(0)
        fp = (probs * (1 - targets)).sum(0)
        fn = ((1 - probs) * targets).sum(0)
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-6)
        return 0.5 * self.bce(logits, targets) + 0.5 * (1 - f1.mean())


aug_train = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.OneOf([A.ImageCompression(60, 100), A.ISONoise()], p=0.4),
    A.OneOf([A.RandomBrightnessContrast(), A.HueSaturationValue()], p=0.3),
    A.ShiftScaleRotate(0.05, 0.1, 15, p=0.4),
    A.CoarseDropout(6, 32, 32, p=0.3),
    ToTensorV2()
])
aug_val = A.Compose([ToTensorV2()])


class DFDataset(Dataset):
    def __init__(self, df, root, tf):
        self.df = df; self.root = root; self.tf = tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        path = os.path.join(self.root, f"{row.Id}.jpg")
        try:
            img = cv2.imread(path)
            if img is None: raise ValueError()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            img = np.zeros((256, 256, 3), dtype=np.uint8)
        if self.tf: img = self.tf(image=img)['image']
        return img.float() / 255.0, torch.tensor(row.target_feature, dtype=torch.float32)


def mixup(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0)).to(CONFIG['device'])
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


kf = StratifiedKFold(n_splits=CONFIG['n_folds'], shuffle=True, random_state=CONFIG['seed'])
fold_thresholds = []

print(f"ЗАПУСК ОБУЧЕНИЯ НА {CONFIG['n_folds']} ФОЛДАХ...")

for fold, (train_idx, val_idx) in enumerate(kf.split(train_df, train_df['target_feature'])):
    print(f"\n=== FOLD {fold + 1}/{CONFIG['n_folds']} ===")
    train_sub = train_df.iloc[train_idx]
    val_sub = train_df.iloc[val_idx]

    counts = train_sub['target_feature'].value_counts()
    weights = [1.0 / counts[x] for x in train_sub['target_feature']]
    sampler = WeightedRandomSampler(weights, len(train_sub), replacement=True)

    train_loader = DataLoader(DFDataset(train_sub, DIRS['train'], aug_train),
                              batch_size=CONFIG['batch_size'], sampler=sampler,
                              num_workers=CONFIG['num_workers'], drop_last=True,
                              pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(DFDataset(val_sub, DIRS['train'], aug_val),
                            batch_size=CONFIG['batch_size'], shuffle=False,
                            num_workers=CONFIG['num_workers'], pin_memory=True, persistent_workers=True)

    model = DualStreamNetwork().to(CONFIG['device'])

    if CONFIG['compile_model']:
        try:
            print(" Compiling model...")
            model = torch.compile(model)
        except Exception as e:
            print(f"Compile skip: {e}")

    if GPU_COUNT > 1:
        model = nn.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    criterion = ComboLoss()
    scaler = GradScaler()
    best_f1 = 0.0

    for ep in range(CONFIG['epochs']):
        model.train()
        loss_acc = 0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Ep {ep + 1}", leave=False)

        for i, (x, y) in enumerate(pbar):
            x, y = x.to(CONFIG['device']), y.to(CONFIG['device']).unsqueeze(1)
            with autocast():
                if random.random() < 0.5:
                    x, ya, yb, lam = mixup(x, y)
                    out = model(x)
                    loss = lam * criterion(out, ya) + (1 - lam) * criterion(out, yb)
                else:
                    loss = criterion(model(x), y)

            scaler.scale(loss).backward()
            scaler.step(optimizer);
            scaler.update();
            optimizer.zero_grad()
            loss_acc += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        scheduler.step()

        model.eval()
        probs, targets = [], []
        with torch.no_grad():
            for x, y in val_loader:
                with autocast(): out = model(x.to(CONFIG['device']))
                probs.extend(torch.sigmoid(out).cpu().numpy().flatten())
                targets.extend(y.numpy())

        best_th_f1, best_th = 0, 0.5
        for th in np.linspace(0.2, 0.8, 60):
            scr = f1_score(targets, (np.array(probs) > th).astype(int))
            if scr > best_th_f1: best_th_f1 = scr; best_th = th

        print(f"Ep {ep + 1} | Loss: {loss_acc / len(train_loader):.4f} | F1: {best_th_f1:.4f} (th={best_th:.2f})")

        if best_th_f1 > best_f1:
            best_f1 = best_th_f1
            to_save = model.module if GPU_COUNT > 1 else model
            if hasattr(to_save, '_orig_mod'): to_save = to_save._orig_mod
            torch.save({'model': to_save.state_dict(), 'th': best_th}, f'model_fold{fold}.pth')
            print("   Saved Best")

    fold_thresholds.append(torch.load(f'model_fold{fold}.pth')['th'])
    del model, optimizer, scaler
    torch.cuda.empty_cache()

print("\nГенерация сабмишна...")
test_ids = sorted(
    [int(f.split('.')[0]) for f in os.listdir('data/dataset/test_images') if f.endswith(('.jpg', '.png'))])


class InferDS(Dataset):
    def __init__(self, ids, root, tf):
        self.ids = ids; self.root = root; self.tf = tf

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        fid = self.ids[i]
        src_path = f"data/dataset/test_images/{fid}.jpg"
        if not os.path.exists(src_path): src_path = f"data/dataset/test_images/{fid}.png"
        meta_score = check_genai_traces(src_path) if os.path.exists(src_path) else 0.0
        path = os.path.join(self.root, f"{fid}.jpg")
        if os.path.exists(path):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return self.tf(image=img)['image'].float() / 255.0, meta_score
        return torch.zeros((3, 256, 256)), 0.0


loader = DataLoader(InferDS(test_ids, DIRS['test'], aug_val), batch_size=CONFIG['batch_size'] * 2, shuffle=False,
                    num_workers=CONFIG['num_workers'])
final_probs = np.zeros(len(test_ids))

for fold in range(CONFIG['n_folds']):
    path = f'model_fold{fold}.pth'
    if not os.path.exists(path): continue
    print(f"Загрузка модели Fold {fold + 1}...")
    ckpt = torch.load(path, map_location=CONFIG['device'])

    model = DualStreamNetwork().to(CONFIG['device'])
    if CONFIG['compile_model']:
        try:
            model = torch.compile(model)
        except:
            pass
    if GPU_COUNT > 1: model = nn.DataParallel(model)

    state_dict = ckpt['model']
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    if list(state_dict.keys())[0].startswith('module.') and GPU_COUNT == 1:
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    if GPU_COUNT > 1 and not list(state_dict.keys())[0].startswith('module.'):
        state_dict = {'module.' + k: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    fold_probs = []
    with torch.no_grad():
        for x, metas in tqdm(loader, desc=f"Infer Fold {fold}"):
            x = x.to(CONFIG['device'])
            p1 = torch.sigmoid(model(x))
            p2 = torch.sigmoid(model(torch.flip(x, [3]))) if CONFIG['tta'] else p1
            fold_probs.extend(((p1 + p2) / 2.0).cpu().numpy().flatten())
    final_probs += np.array(fold_probs)

final_probs /= CONFIG['n_folds']
avg_th = np.mean(fold_thresholds)
predictions = (final_probs > avg_th).astype(int)
submission = pd.DataFrame({'Id': test_ids, 'target_feature': predictions})
submission.to_csv('submission.csv', index=False)
print(f" Готово! Файл submission.csv создан. Порог: {avg_th}")