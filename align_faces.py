import os
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image

SRC_TRAIN = 'data/dataset/train_images'
SRC_TEST = 'data/dataset/test_images'
if not os.path.exists(SRC_TRAIN): SRC_TRAIN = 'data/faces_train'
if not os.path.exists(SRC_TEST): SRC_TEST = 'data/faces_test'

DST_ROOT = 'data/faces_384'
IMG_SIZE = 384
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"FACE ALIGNMENT V42 (SMART CROP) | Device: {DEVICE}")


class ImageDataset(Dataset):
    def __init__(self, root):
        self.files = [os.path.join(root, f) for f in os.listdir(root) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self): return len(self.files)

    def __getitem__(self, i): return self.files[i]


def align_dataset(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)

    mtcnn = MTCNN(keep_all=False, select_largest=True, margin=40, post_process=False, device=DEVICE)

    dataset = ImageDataset(src_dir)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=lambda x: x)

    print(f" Processing {src_dir} -> {dst_dir}...")

    for paths in tqdm(loader):
        batch_imgs = []
        valid_paths = []

        # 1. Загрузка в память
        for p in paths:
            try:
                img = Image.open(p).convert('RGB')
                batch_imgs.append(img)
                valid_paths.append(p)
            except:
                pass

        if not batch_imgs: continue

        # 2. Детекция батчем
        try:
            boxes, probs = mtcnn.detect(batch_imgs)
        except:
            boxes = [None] * len(batch_imgs)

        # 3. Обработка и сохранение
        for i, img_path in enumerate(valid_paths):
            fname = os.path.basename(img_path)
            save_path = os.path.join(dst_dir, fname)

            img = batch_imgs[i]
            box = boxes[i]

            if box is not None:
                b = box[0].astype(int)
                x1, y1, x2, y2 = b
                w_box, h_box = x2 - x1, y2 - y1

                # Расширяем кроп на 20% для контекста
                x1 = max(0, x1 - int(w_box * 0.2))
                y1 = max(0, y1 - int(h_box * 0.2))
                x2 = min(img.width, x2 + int(w_box * 0.2))
                y2 = min(img.height, y2 + int(h_box * 0.2))

                try:
                    face = img.crop((x1, y1, x2, y2))
                except:
                    face = img
            else:
                w, h = img.size
                s = min(w, h)
                left = (w - s) // 2
                top = (h - s) // 2
                face = img.crop((left, top, left + s, top + s))

            face = face.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BICUBIC)
            face.save(save_path, quality=95)


if __name__ == '__main__':
    align_dataset(SRC_TRAIN, os.path.join(DST_ROOT, 'train'))
    align_dataset(SRC_TEST, os.path.join(DST_ROOT, 'test'))

    print("ALIGNMENT COMPLETE. Ready for V42 Training.")
