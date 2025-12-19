import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b4 as NativeMBConv
from PIL import Image

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEST_DIR = 'data/faces_384/test'
SUBMISSION_PATH = 'submission.csv'
WEIGHTS_FILE = 'submission_weights.pth'


class TitanNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = NativeMBConv(weights=None)
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1792, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        return self.backbone(x)


class TestDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        path = os.path.join(self.root_dir, fname)
        try:
            img = Image.open(path).convert('RGB')
            img = self.transform(img)
            return img, fname
        except:
            return torch.zeros((3, 384, 384)), fname


def main():
    print("Starting inference TitanNet V42 (Native MBConv)...")

    model = TitanNet().to(DEVICE)
    model.eval()

    if not os.path.exists(WEIGHTS_FILE):
        print(f"Error: Weights file '{WEIGHTS_FILE}' not found.")
        print("Please download it from the link in README and place it in the root directory.")
        return

    try:
        state_dict = torch.load(WEIGHTS_FILE, map_location=DEVICE)
        model.backbone.load_state_dict(state_dict)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Load Error: {e}")
        return

    if not os.path.exists(TEST_DIR):
        pd.DataFrame({'filename': [], 'prediction': []}).to_csv(SUBMISSION_PATH, index=False)
        return

    dataset = TestDataset(TEST_DIR)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

    predictions = []
    filenames = []

    with torch.no_grad():
        for imgs, fnames in loader:
            imgs = imgs.to(DEVICE)
            out = model(imgs).view(-1)
            out_flip = model(torch.flip(imgs, [3])).view(-1)

            probs = torch.sigmoid((out + out_flip) / 2)

            predictions.extend(probs.cpu().numpy())
            filenames.extend(fnames)

    df = pd.DataFrame({'filename': filenames, 'prediction': predictions})
    df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Inference done. Saved to {SUBMISSION_PATH}")


if __name__ == '__main__':
    main()