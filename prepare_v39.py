import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score

CONFIG = {
    'batch_size': 128,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'num_workers': 8,
    'pseudo_threshold': 0.995,
    'use_amp': torch.cuda.is_available()
}

print(f"POST-PROCESSING & PSEUDO-LABELING | Device: {CONFIG['device']}")


def load_model_weights(model, path):
    try:
        state = torch.load(path, map_location=CONFIG['device'], weights_only=False)
        if 'model' in state:
            state = state['model']
        elif 'backbone' in state:
            state = state['backbone']
        new_state = {k.replace('module.', ''): v for k, v in state.items()}
        model.load_state_dict(new_state, strict=False)
        return True
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return False


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
    print("Loading V38 Models...")
    models = []
    for i in range(5):
        bk_path = f'titan_backbone_fold{i}.pth'
        hd_path = f'titan_head_fold{i}.pth'
        if os.path.exists(bk_path):
            bk = TitanNet().to(CONFIG['device']).eval()
            hd = ArcFaceHead().to(CONFIG['device']).eval()
            if load_model_weights(bk, bk_path) and load_model_weights(hd, hd_path):
                models.append((bk, hd))

    if not models: exit(" No models found! Wait for V38 to finish.")
    print(f"Loaded {len(models)} models.")

    test_dir = 'data/dataset/test_images'
    if not os.path.exists(test_dir): test_dir = 'data/faces_test'
    test_files = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
    test_ids = sorted([int(f.split('.')[0]) for f in test_files])

    loader = DataLoader(InferDS(test_ids, test_dir), batch_size=CONFIG['batch_size'],
                        shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=True)

    print("Running Inference...")
    pseudo_data = []
    final_probs = []
    ids_list = []

    from torch.cuda.amp import autocast

    with torch.no_grad():
        for x, ids in tqdm(loader):
            x = x.to(CONFIG['device']).float().div(255.0)
            batch_probs = torch.zeros(x.shape[0], device=CONFIG['device'])

            with autocast(enabled=CONFIG['use_amp']):
                for bk, hd in models:
                    p1 = torch.softmax(hd(bk(x), None, use_arcface=True), dim=1)[:, 1]
                    p2 = torch.softmax(hd(bk(torch.flip(x, [3])), None, use_arcface=True), dim=1)[:, 1]
                    batch_probs += (p1 + p2) / 2

            batch_probs /= len(models)

            final_probs.extend(batch_probs.cpu().numpy())
            ids_list.extend(ids.numpy())

    thresholds = []
    for i in range(5):
        try:
            meta = torch.load(f'titan_meta_fold{i}.pth', weights_only=False)
            thresholds.append(meta['th'])
        except:
            pass

    avg_th = np.mean(thresholds) if thresholds else 0.5
    print(f" Average Optimal Threshold: {avg_th:.4f}")

    sub = pd.DataFrame({'Id': ids_list, 'target_feature': (np.array(final_probs) > avg_th).astype(int)})
    sub.to_csv('submission_v38.csv', index=False)
    print("Saved submission_v38.csv (Base Result)")

    print(" Selecting Pseudo-Labels...")
    count = 0
    for pid, prob in zip(ids_list, final_probs):
        if prob > CONFIG['pseudo_threshold']:
            pseudo_data.append({'Id': pid, 'target_feature': 1, 'is_pseudo': True})
            count += 1
        elif prob < (1 - CONFIG['pseudo_threshold']):
            pseudo_data.append({'Id': pid, 'target_feature': 0, 'is_pseudo': True})
            count += 1

    print(f"Selected {count} confident samples (Confidence > {CONFIG['pseudo_threshold'] * 100}%)")

    orig_df = pd.read_csv('data/dataset/train_solution.csv')
    if 'id' in orig_df.columns: orig_df.rename(columns={'id': 'Id'}, inplace=True)
    if 'target' in orig_df.columns: orig_df.rename(columns={'target': 'target_feature'}, inplace=True)
    orig_df['is_pseudo'] = False

    pseudo_df = pd.DataFrame(pseudo_data)
    final_df = pd.concat([orig_df[['Id', 'target_feature', 'is_pseudo']], pseudo_df])
    final_df.to_csv('train_sniper.csv', index=False)
    print(f" Saved 'train_sniper.csv' for V39 tuning!")