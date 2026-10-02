
import glob
import os
import re

import mlflow
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = "/kaggle/input/competitions/ultrasound-nerve-segmentation"
H, W = 96, 128      # images are shrunk to this size
EPOCHS = 10
BATCH_SIZE = 32
SEED = 0
RUN_NAME = "step1_baseline"     # the name of this run in the MLflow table

torch.manual_seed(SEED)
np.random.seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"


##### 0. Start the MLflow run
# The history is kept in the file mlflow.db (plus a folder mlruns for saved code).
# Both are created in the folder you run from.

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nerve_segmentation")
mlflow.start_run(run_name=RUN_NAME)
mlflow.log_params({"step": 1, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "image_size": f"{H}x{W}",
                   "oversample": False, "pixel_weight": 1.0, "dice_loss": False})


##### 1. Load the images and masks
# Training files are named <patient>_<frame>.tif, with a mask <patient>_<frame>_mask.tif.

images, masks, patients = [], [], []
for path in sorted(glob.glob(os.path.join(DATA_DIR, "**", "*.tif"), recursive=True)):
    match = re.match(r"^(\d+)_(\d+)\.tif$", os.path.basename(path))
    mask_path = path[:-4] + "_mask.tif"
    if match is None or not os.path.exists(mask_path):
        continue
    img = Image.open(path).convert("L").resize((W, H), Image.BILINEAR)
    msk = Image.open(mask_path).convert("L").resize((W, H), Image.NEAREST)
    images.append(np.asarray(img, dtype=np.float32) / 255.0)
    masks.append((np.asarray(msk) > 127).astype(np.float32))
    patients.append(int(match.group(1)))

X = np.stack(images)
Y = np.stack(masks)
patients = np.array(patients)
has_nerve = Y.sum(axis=(1, 2)) > 0
print(f"{len(X)} frames, {len(set(patients))} patients, nerve in {has_nerve.mean():.0%} of frames")


##### 2. Split by patient
# Frames of one patient look almost the same.
# So a patient must be entirely in train or entirely in test.

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx, test_idx = next(splitter.split(X, groups=patients))
print(f"train: {len(train_idx)} frames, test: {len(test_idx)} frames")


##### 3. The model: a small U-Net

def block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1), nn.ReLU(),
        nn.Conv2d(c_out, c_out, 3, padding=1), nn.ReLU(),
    )

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.enc1 = block(1, 16)
        self.enc2 = block(16, 32)
        self.middle = block(32, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = block(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = block(32, 16)
        self.out = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)                                  # full size
        e2 = self.enc2(self.pool(e1))                      # half size
        m = self.middle(self.pool(e2))                     # quarter size
        d2 = self.dec2(torch.cat([self.up2(m), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)                                # one score per pixel

model = UNet().to(device)


##### 4. Train
# Loss: for every pixel, "nerve or not". Plain binary cross-entropy.

train_set = TensorDataset(torch.from_numpy(X[train_idx]).unsqueeze(1),
                          torch.from_numpy(Y[train_idx]).unsqueeze(1))
loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(EPOCHS):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        loss = loss_fn(model(xb), yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(xb)
    print(f"epoch {epoch + 1}/{EPOCHS}  loss {total / len(train_idx):.4f}")
    mlflow.log_metric("train_loss", total / len(train_idx), step=epoch + 1)


##### 5. Predict on the test patients

model.eval()
pred_masks = []
with torch.no_grad():
    for start in range(0, len(test_idx), 64):
        xb = torch.from_numpy(X[test_idx[start:start + 64]]).unsqueeze(1).to(device)
        prob = torch.sigmoid(model(xb))[:, 0].cpu().numpy()
        pred_masks.append(prob > 0.5)
pred_masks = np.concatenate(pred_masks)


##### 6. Score
# Dice per frame = 2 * overlap / (predicted pixels + true pixels).
# An empty prediction on an empty frame counts as 1.

true_masks = Y[test_idx] > 0.5
overlap = (pred_masks & true_masks).sum(axis=(1, 2))
total = pred_masks.sum(axis=(1, 2)) + true_masks.sum(axis=(1, 2))
dice = np.where(total == 0, 1.0, 2 * overlap / np.maximum(total, 1))

test_has_nerve = has_nerve[test_idx]
predicted_nerve = pred_masks.any(axis=(1, 2))
found = predicted_nerve[test_has_nerve].mean()
kept_empty = (~predicted_nerve[~test_has_nerve]).mean()
always_empty = (~test_has_nerve).mean()

print("\n--- test results ---")
print(f"mean Dice, all frames:       {dice.mean():.3f}")
print(f"mean Dice, nerve frames:     {dice[test_has_nerve].mean():.3f}")
print(f"nerve frames found:          {found:.0%}")
print(f"empty frames kept empty:     {kept_empty:.0%}")
print(f"'always empty' would score:  {always_empty:.3f}   (all frames)")


##### 7. Save the results in MLflow
# Same metric names in every step, so the runs line up in one table.

mlflow.log_metrics({"dice_all": float(dice.mean()),
                    "dice_nerve_frames": float(dice[test_has_nerve].mean()),
                    "nerve_found": float(found),
                    "empty_kept_empty": float(kept_empty),
                    "always_empty_dice": float(always_empty)})
try:
    mlflow.log_artifact(__file__)       # saves a copy of this code with the run
except NameError:
    pass                                # in a notebook there is no file to copy
mlflow.end_run()
print("saved run to MLflow:", RUN_NAME)
