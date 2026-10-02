##### Step 5: better scoring and clean-up
# Same file as step 4. Changes are marked with ">>> STEP 5":
#   1. A small group of validation patients is held out. It is only used to choose thresholds.
#   2. Balanced score: nerve frames and empty frames each count for half.
#   3. Three thresholds are chosen on the validation patients instead of being fixed at 0.5:
#      the detection threshold, the pixel threshold and the minimum blob size.
#   4. Clean-up: keep only the largest blob in each mask, and drop tiny blobs.
# Why: plain mean Dice rewards empty answers. "Always empty" already scores about 0.49.
# The balanced score gives "always empty" exactly 0.5, so a model must find nerves to beat it.
# Note: the validation patients are not used for training, so this step trains on slightly fewer
# patients than step 4. Compare the two rows printed in this run ("default" and "tuned").
# That comparison uses the same trained model, so it shows what scoring and clean-up add.
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
from scipy import ndimage
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

DATA_DIR = "/kaggle/input/competitions/ultrasound-nerve-segmentation"
H, W = 96, 128      # images are shrunk to this size
EPOCHS = 10
BATCH_SIZE = 32
SEED = 0
PIXEL_WEIGHT = 5.0                  # from step 3
HEAD_WEIGHT = 0.5                   # from step 4
VAL_FRACTION = 0.15                 # >>> STEP 5: share of training patients kept for choosing thresholds
PRES_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]    # >>> STEP 5: values to try for the detection threshold
PIX_GRID = [0.3, 0.4, 0.5, 0.6]                          # values to try for the pixel threshold
AREA_GRID = [0, 25, 50, 100, 200, 300, 400]              # smallest allowed blob, in pixels on the 96x128 grid
RUN_NAME = "step5_scoring_cleanup"  # the name of this run in the MLflow table

torch.manual_seed(SEED)
np.random.seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"


##### 0. Start the MLflow run
# The history is kept in the file mlflow.db (plus a folder mlruns for saved code).
# Both are created in the folder you run from.

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nerve_segmentation")
mlflow.start_run(run_name=RUN_NAME)
mlflow.log_params({"step": 5, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "image_size": f"{H}x{W}",
                   "oversample": True, "pixel_weight": PIXEL_WEIGHT, "dice_loss": True,
                   "detection_head": True, "head_weight": HEAD_WEIGHT,
                   "val_fraction": VAL_FRACTION, "tuned_thresholds": True, "largest_blob_only": True})


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
# Frames of one patient look almost the same, so a patient is entirely in one group.

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
dev_idx, test_idx = next(splitter.split(X, groups=patients))     # same test patients as steps 1 to 4

# >>> STEP 5: hold out a validation group (whole patients) from the non-test patients.
# Thresholds are chosen here. They must never be chosen on the test patients.
val_splitter = GroupShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=SEED)
tr, va = next(val_splitter.split(dev_idx, groups=patients[dev_idx]))
train_idx, val_idx = dev_idx[tr], dev_idx[va]

assert not (set(patients[train_idx]) & set(patients[val_idx])), "a patient is in train and validation"
assert not (set(patients[dev_idx]) & set(patients[test_idx])), "a patient is in train and test"
for name, ids in [("train", train_idx), ("validation", val_idx), ("test", test_idx)]:
    print(f"{name:10s}: {len(ids):5d} frames, {len(set(patients[ids])):2d} patients, nerve in {has_nerve[ids].mean():.0%}")


##### 3. The model: a small U-Net with two outputs (same as step 4)

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
        self.presence = nn.Linear(64 * 2, 1)               # the detection head

    def forward(self, x):
        e1 = self.enc1(x)                                  # full size
        e2 = self.enc2(self.pool(e1))                      # half size
        m = self.middle(self.pool(e2))                     # quarter size
        d2 = self.dec2(torch.cat([self.up2(m), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        pooled = torch.cat([m.mean(dim=(2, 3)), m.amax(dim=(2, 3))], dim=1)
        return self.out(d1), self.presence(pooled)[:, 0]

model = UNet().to(device)


##### 4. Train (same as step 4, but on the smaller training group)

train_set = TensorDataset(torch.from_numpy(X[train_idx]).unsqueeze(1),
                          torch.from_numpy(Y[train_idx]).unsqueeze(1))

train_has_nerve = has_nerve[train_idx]
nerve_weight = (~train_has_nerve).sum() / train_has_nerve.sum()
weights = np.where(train_has_nerve, nerve_weight, 1.0)
sampler = WeightedRandomSampler(torch.from_numpy(weights).double(),
                                num_samples=len(train_idx), replacement=True)
loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler)

pixel_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(PIXEL_WEIGHT, device=device))
presence_loss_fn = nn.BCEWithLogitsLoss()

def dice_loss_fn(logits, target):
    prob = torch.sigmoid(logits)
    overlap = (prob * target).sum(dim=(1, 2, 3))
    total = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return 1 - ((2 * overlap + 1) / (total + 1)).mean()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(EPOCHS):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        seg_logits, pres_logit = model(xb)
        frame_label = yb.amax(dim=(1, 2, 3))
        loss = (pixel_loss_fn(seg_logits, yb) + dice_loss_fn(seg_logits, yb)
                + HEAD_WEIGHT * presence_loss_fn(pres_logit, frame_label))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(xb)
    print(f"epoch {epoch + 1}/{EPOCHS}  loss {total / len(train_idx):.4f}")
    mlflow.log_metric("train_loss", total / len(train_idx), step=epoch + 1)


##### 5. Predict probabilities for any group of frames

def predict(ids):
    model.eval()
    seg, pres = [], []
    with torch.no_grad():
        for start in range(0, len(ids), 64):
            xb = torch.from_numpy(X[ids[start:start + 64]]).unsqueeze(1).to(device)
            seg_logits, pres_logit = model(xb)
            seg.append(torch.sigmoid(seg_logits)[:, 0].cpu().numpy())
            pres.append(torch.sigmoid(pres_logit).cpu().numpy())
    return np.concatenate(seg), np.concatenate(pres)


##### 6. Turn probabilities into masks (the clean-up), and score them
# >>> STEP 5: make_masks does four things, in this order:
#   a) the head says "no nerve" (probability below pres_thr)  -> empty mask
#   b) pixels above pix_thr become nerve
#   c) keep only the largest blob (a frame holds one nerve)
#   d) a blob smaller than min_area is probably a false alarm  -> empty mask

def make_masks(seg_probs, pres_probs, pres_thr, pix_thr, min_area, largest_only):
    masks = seg_probs > pix_thr
    for i in range(len(masks)):
        if pres_probs[i] < pres_thr:
            masks[i] = False
            continue
        if largest_only and masks[i].any():
            blobs, n_blobs = ndimage.label(masks[i])
            if n_blobs > 1:
                sizes = ndimage.sum(masks[i], blobs, range(1, n_blobs + 1))
                masks[i] = blobs == (np.argmax(sizes) + 1)
        if masks[i].sum() < min_area:
            masks[i] = False
    return masks

# Dice per frame. An empty prediction on an empty frame counts as 1.
def frame_dice(pred_masks, true_masks):
    overlap = (pred_masks & true_masks).sum(axis=(1, 2))
    total = pred_masks.sum(axis=(1, 2)) + true_masks.sum(axis=(1, 2))
    return np.where(total == 0, 1.0, 2 * overlap / np.maximum(total, 1))

# >>> STEP 5: balanced score = half the Dice of nerve frames + half the Dice of empty frames.
# "Always empty" gets 0 on nerve frames and 1 on empty frames, so exactly 0.5.
def balanced_score(dice, frame_has_nerve):
    return 0.5 * dice[frame_has_nerve].mean() + 0.5 * dice[~frame_has_nerve].mean()


##### 7. Choose the three thresholds on the validation patients
# One threshold at a time: try every value in its grid, keep the one with the best balanced score.

val_seg, val_pres = predict(val_idx)
val_true = Y[val_idx] > 0.5
val_has = has_nerve[val_idx]
assert val_has.any() and (~val_has).any(), "validation group needs nerve frames and empty frames"

def val_score(pres_thr, pix_thr, min_area):
    val_masks = make_masks(val_seg, val_pres, pres_thr, pix_thr, min_area, largest_only=True)
    return balanced_score(frame_dice(val_masks, val_true), val_has)

best = {"pres_thr": 0.5, "pix_thr": 0.5, "min_area": 0}
for name, grid in [("pres_thr", PRES_GRID), ("pix_thr", PIX_GRID), ("min_area", AREA_GRID)]:
    scores = []
    for value in grid:
        trial = {**best, name: value}
        scores.append(val_score(**trial))
    best[name] = grid[int(np.argmax(scores))]
    print(f"chosen {name:9s} = {best[name]}   (validation balanced score {max(scores):.3f})")


##### 8. Test results
# The test patients are used once, with the thresholds chosen above.

test_seg, test_pres = predict(test_idx)
test_true = Y[test_idx] > 0.5
test_has = has_nerve[test_idx]
always_empty = (~test_has).mean()

def report(pred_masks):
    d = frame_dice(pred_masks, test_true)
    found = pred_masks.any(axis=(1, 2))
    return {"dice_all": d.mean(),
            "dice_nerve_frames": d[test_has].mean(),
            "nerve_found": found[test_has].mean(),
            "empty_kept_empty": (~found[~test_has]).mean(),
            "balanced": balanced_score(d, test_has)}

# default = what step 4 did: head at 0.5, pixels at 0.5, no clean-up
default = report(make_masks(test_seg, test_pres, 0.5, 0.5, 0, largest_only=False))
tuned = report(make_masks(test_seg, test_pres, **best, largest_only=True))

print("\n--- test results ---")
print(f"always empty:  balanced 0.500   Dice all {always_empty:.3f}")
for name, m in [("default", default), ("tuned", tuned)]:
    print(f"{name:13s}  balanced {m['balanced']:.3f}   Dice all {m['dice_all']:.3f}   nerve frames {m['dice_nerve_frames']:.3f}   "
          f"found {m['nerve_found']:.0%}   empty kept empty {m['empty_kept_empty']:.0%}")


##### 9. Save the results in MLflow
# Same metric names in every step, so the runs line up in one table.
# The saved numbers are the "tuned" ones. The chosen thresholds are saved as settings.

mlflow.log_params({"tuned_pres_thr": best["pres_thr"], "tuned_pix_thr": best["pix_thr"],
                   "tuned_min_area": best["min_area"]})
for name, value in tuned.items():
    mlflow.log_metric(name, float(value))
mlflow.log_metric("always_empty_dice", float(always_empty))
mlflow.log_metric("default_balanced", float(default["balanced"]))
try:
    mlflow.log_artifact(__file__)       # saves a copy of this code with the run
except NameError:
    pass                                # in a notebook there is no file to copy
mlflow.end_run()
print("saved run to MLflow:", RUN_NAME)
