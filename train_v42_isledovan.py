import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import efficientnet_b4 as NativeMBConv
from PIL import Image

CONFIG = {
    'lr': 1e-3,
    'epochs': 20,
    'batch_size': 32,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}


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


class DiskDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.root_dir, self.files[idx])
        img = Image.open(path).convert('RGB')
        return self.transform(img), torch.tensor(0.0)


def train():
    model = TitanNet().to(CONFIG['device'])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    criterion = nn.BCEWithLogitsLoss()

    print("Initializing TitanNet (MBConv) training pipeline...")

    if os.path.exists('data/train'):
        ds = DiskDataset('data/train')
        loader = DataLoader(ds, batch_size=CONFIG['batch_size'], shuffle=True)

        for epoch in range(CONFIG['epochs']):
            model.train()
            for x, y in loader:
                x, y = x.to(CONFIG['device']), y.to(CONFIG['device'])
                optimizer.zero_grad()
                out = model(x).view(-1)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch + 1} finished.")


if __name__ == '__main__':
    train()
