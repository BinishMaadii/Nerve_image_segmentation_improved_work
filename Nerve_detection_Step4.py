##### Step 4: add a detection head ("is there a nerve in this frame?")
# Same file as step 3. Oversampling, pixel weight and Dice loss stay on.
# Changes are marked with ">>> STEP 4":
#   1. The model gets a second output: one yes/no score for the whole frame.
#   2. The loss gets a third term that trains this score.
#   3. At prediction time, if the head says "no nerve", the mask is set to empty.
# Why: on empty frames the pixel output often draws small false blobs.
# One decision per frame is easier than about 12,000 decisions per pixel.
# The run is saved with MLflow, so the steps can be compared later.
# Needs:  pip install mlflow   (in a Kaggle notebook:  !pip install mlflow)
# Data: Kaggle "Ultrasound Nerve Segmentation" (neck ultrasound, brachial plexus).

import glob
import os
import re

import mlflow
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

DATA_DIR = "/kaggle/input/competitions/ultrasound-nerve-segmentation"
H, W = 96, 128      # images are shrunk to this size
EPOCHS = 10
BATCH_SIZE = 32
SEED = 0
PIXEL_WEIGHT = 5.0                  # from step 3
HEAD_WEIGHT = 0.5                   # >>> STEP 4: how much the detection loss counts
PRESENCE_THR = 0.5                  # >>> STEP 4: head probability below this = "no nerve"
RUN_NAME = "step4_detection_head"   # the name of this run in the MLflow table

torch.manual_seed(SEED)
np.random.seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"


##### 0. Start the MLflow run
# The history is kept in the file mlflow.db (plus a folder mlruns for saved code).
# Both are created in the folder you run from.

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nerve_segmentation")
mlflow.start_run(run_name=RUN_NAME)
mlflow.log_params({"step": 4, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "image_size": f"{H}x{W}",
                   "oversample": True, "pixel_weight": PIXEL_WEIGHT, "dice_loss": True,
                   "detection_head": True, "head_weight": HEAD_WEIGHT, "presence_thr": PRESENCE_THR})


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


##### 3. The model: a small U-Net with two outputs

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
        # >>> STEP 4: the detection head.
        # It reads the deepest layer (64 channels). For each channel it takes the average and
        # the strongest value over the whole image. That gives 128 numbers, which become 1 score.
        self.presence = nn.Linear(64 * 2, 1)

    def forward(self, x):
        e1 = self.enc1(x)                                  # full size
        e2 = self.enc2(self.pool(e1))                      # half size
        m = self.middle(self.pool(e2))                     # quarter size
        d2 = self.dec2(torch.cat([self.up2(m), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        pooled = torch.cat([m.mean(dim=(2, 3)), m.amax(dim=(2, 3))], dim=1)    # >>> STEP 4
        return self.out(d1), self.presence(pooled)[:, 0]   # >>> STEP 4: two outputs now

model = UNet().to(device)


##### 4. Train

train_set = TensorDataset(torch.from_numpy(X[train_idx]).unsqueeze(1),
                          torch.from_numpy(Y[train_idx]).unsqueeze(1))

# Step 2 (kept): nerve frames are drawn more often, so a batch is about half nerve, half empty.
train_has_nerve = has_nerve[train_idx]
nerve_weight = (~train_has_nerve).sum() / train_has_nerve.sum()
weights = np.where(train_has_nerve, nerve_weight, 1.0)
sampler = WeightedRandomSampler(torch.from_numpy(weights).double(),
                                num_samples=len(train_idx), replacement=True)
loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler)

# Step 3 (kept): pixel weight and Dice term.
pixel_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(PIXEL_WEIGHT, device=device))

def dice_loss_fn(logits, target):
    prob = torch.sigmoid(logits)
    overlap = (prob * target).sum(dim=(1, 2, 3))
    total = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return 1 - ((2 * overlap + 1) / (total + 1)).mean()

# >>> STEP 4: loss for the detection head. Plain yes/no loss, one decision per frame.
presence_loss_fn = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(EPOCHS):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        seg_logits, pres_logit = model(xb)                       # >>> STEP 4: two outputs
        frame_label = yb.amax(dim=(1, 2, 3))                     # >>> STEP 4: 1 if the frame has any nerve pixel
        loss = (pixel_loss_fn(seg_logits, yb) + dice_loss_fn(seg_logits, yb)
                + HEAD_WEIGHT * presence_loss_fn(pres_logit, frame_label))    # >>> STEP 4: third term
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(xb)
    print(f"epoch {epoch + 1}/{EPOCHS}  loss {total / len(train_idx):.4f}")
    mlflow.log_metric("train_loss", total / len(train_idx), step=epoch + 1)


##### 5. Predict on the test patients
# We keep the probabilities, so the mask can be made with and without the head.

model.eval()
seg_probs, pres_probs = [], []
with torch.no_grad():
    for start in range(0, len(test_idx), 64):
        xb = torch.from_numpy(X[test_idx[start:start + 64]]).unsqueeze(1).to(device)
        seg_logits, pres_logit = model(xb)
        seg_probs.append(torch.sigmoid(seg_logits)[:, 0].cpu().numpy())
        pres_probs.append(torch.sigmoid(pres_logit).cpu().numpy())
seg_probs = np.concatenate(seg_probs)       # one probability per pixel, shape (frames, 96, 128)
pres_probs = np.concatenate(pres_probs)     # one probability per frame

masks_without_head = seg_probs > 0.5
# >>> STEP 4: if the head says "no nerve" (probability below PRESENCE_THR), the whole mask is emptied.
masks_with_head = masks_without_head & (pres_probs > PRESENCE_THR)[:, None, None]


##### 6. Score
# Dice per frame = 2 * overlap / (predicted pixels + true pixels).
# An empty prediction on an empty frame counts as 1.

true_masks = Y[test_idx] > 0.5
test_has_nerve = has_nerve[test_idx]
always_empty = (~test_has_nerve).mean()

def score(pred_masks):
    overlap = (pred_masks & true_masks).sum(axis=(1, 2))
    total = pred_masks.sum(axis=(1, 2)) + true_masks.sum(axis=(1, 2))
    dice = np.where(total == 0, 1.0, 2 * overlap / np.maximum(total, 1))
    predicted_nerve = pred_masks.any(axis=(1, 2))
    return {"dice_all": dice.mean(),
            "dice_nerve_frames": dice[test_has_nerve].mean(),
            "nerve_found": predicted_nerve[test_has_nerve].mean(),
            "empty_kept_empty": (~predicted_nerve[~test_has_nerve]).mean()}

scores_without = score(masks_without_head)
scores_with = score(masks_with_head)
head_accuracy = ((pres_probs > PRESENCE_THR) == test_has_nerve).mean()    # how often the head alone is right

print("\n--- test results ---")
for name, m in [("without head", scores_without), ("with head", scores_with)]:
    print(f"{name:13s} Dice all {m['dice_all']:.3f}   nerve frames {m['dice_nerve_frames']:.3f}   "
          f"found {m['nerve_found']:.0%}   empty kept empty {m['empty_kept_empty']:.0%}")
print(f"head alone is right on {head_accuracy:.0%} of frames")
print(f"'always empty' would score:  {always_empty:.3f}   (all frames)")


##### 7. Save the results in MLflow
# Same metric names in every step, so the runs line up in one table.
# The saved numbers are the "with head" ones.

for name, value in scores_with.items():
    mlflow.log_metric(name, float(value))
mlflow.log_metric("always_empty_dice", float(always_empty))
mlflow.log_metric("head_accuracy", float(head_accuracy))
try:
    mlflow.log_artifact(__file__)       # saves a copy of this code with the run
except NameError:
    pass                                # in a notebook there is no file to copy
mlflow.end_run()
print("saved run to MLflow:", RUN_NAME)
