##### Step 8: train the 5 fold models 
#
#   3. The patient split is saved (split.npz), so all steps use exactly the same test patients.
#   4. The held-out fold is also scored after every epoch (val_loss), only to draw learning curves.
#      It is not used to pick epochs or models.
#   5. The U-Net can return its deepest feature map (return_features=True). Step 10 needs it.
#      The weights and the normal output are unchanged.



import glob
import json
import os
import re
import time

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

DATA_DIR = os.environ.get("DATA_DIR", "/kaggle/input/competitions/ultrasound-nerve-segmentation")
OUT_DIR = os.environ.get("OUT_DIR", "outputs")
H, W = 96, 128      # images are shrunk to this size
EPOCHS = int(os.environ.get("EPOCHS", 10))
BATCH_SIZE = 32
SEED = 0
PIXEL_WEIGHT = 5.0                  # from step 3
HEAD_WEIGHT = 0.5                   # from step 4
N_FOLDS = int(os.environ.get("N_FOLDS", 5))
RUN_NAME = "step8_train"

for sub in ["models", "tables", "plots"]:
    os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = "cuda" if torch.cuda.is_available() else "cpu"


##### 0. Start the MLflow run

mlflow.set_tracking_uri(f"sqlite:///{os.path.join(OUT_DIR, 'mlflow.db')}")
mlflow.set_experiment("nerve_segmentation")
mlflow.start_run(run_name=RUN_NAME)
mlflow.log_params({"step": 8, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "image_size": f"{H}x{W}",
                   "pixel_weight": PIXEL_WEIGHT, "head_weight": HEAD_WEIGHT, "n_folds": N_FOLDS, "seed": SEED})


##### 1. Load the images and masks (or read them from the cache)
# Training files are named <patient>_<frame>.tif, with a mask <patient>_<frame>_mask.tif.
# The cache keeps the resized images as uint8 (0 to 255), so dividing by 255 gives the same numbers as before.

cache_path = os.path.join(OUT_DIR, "data_cache.npz")
if os.path.exists(cache_path):
    cache = np.load(cache_path)
    X8, Y8, patients, frame_ids = cache["X"], cache["Y"], cache["patients"], cache["frame_ids"]
    print(f"read the cache {cache_path}")
else:
    images, masks, patients, frame_ids = [], [], [], []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "**", "*.tif"), recursive=True)):
        match = re.match(r"^(\d+)_(\d+)\.tif$", os.path.basename(path))
        mask_path = path[:-4] + "_mask.tif"
        if match is None or not os.path.exists(mask_path):
            continue
        img = Image.open(path).convert("L").resize((W, H), Image.BILINEAR)
        msk = Image.open(mask_path).convert("L").resize((W, H), Image.NEAREST)
        images.append(np.asarray(img, dtype=np.uint8))
        masks.append((np.asarray(msk) > 127).astype(np.uint8))
        patients.append(int(match.group(1)))
        frame_ids.append(int(match.group(2)))
    assert len(images) > 0, f"no <patient>_<frame>.tif files with masks found under {DATA_DIR}"
    X8, Y8 = np.stack(images), np.stack(masks)
    patients, frame_ids = np.array(patients), np.array(frame_ids)
    np.savez(cache_path, X=X8, Y=Y8, patients=patients, frame_ids=frame_ids)
    print(f"saved the cache {cache_path}")

X = X8.astype(np.float32) / 255.0
Y = Y8.astype(np.float32)
has_nerve = Y.sum(axis=(1, 2)) > 0
print(f"{len(X)} frames, {len(set(patients))} patients, nerve in {has_nerve.mean():.0%} of frames")


##### 2. Split by patient
# First the test patients are set aside (same call as steps 1 to 7, so the same patients).
# The other patients ("dev") are split into N_FOLDS groups. GroupKFold keeps a patient in one group.

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
dev_idx, test_idx = next(splitter.split(X, groups=patients))
assert not (set(patients[dev_idx]) & set(patients[test_idx])), "a patient is in dev and test"
folds = list(GroupKFold(n_splits=N_FOLDS).split(dev_idx, groups=patients[dev_idx]))
fold_of_dev = np.zeros(len(dev_idx), dtype=int)         # which fold each dev frame belongs to
for k, (tr, va) in enumerate(folds):
    assert not (set(patients[dev_idx[tr]]) & set(patients[dev_idx[va]])), f"fold {k}: a patient is in train and validation"
    fold_of_dev[va] = k
np.savez(os.path.join(OUT_DIR, "split.npz"), dev_idx=dev_idx, test_idx=test_idx, fold_of_dev=fold_of_dev)
print(f"dev: {len(dev_idx)} frames, {len(set(patients[dev_idx]))} patients   "
      f"test: {len(test_idx)} frames, {len(set(patients[test_idx]))} patients")


##### 3. The model: a small U-Net with two outputs (same weights layout as step 7)
# return_features=True also returns the deepest feature map. Step 10 uses it for Grad-CAM.

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

    def forward(self, x, return_features=False):
        e1 = self.enc1(x)                                  # full size
        e2 = self.enc2(self.pool(e1))                      # half size
        m = self.middle(self.pool(e2))                     # quarter size
        d2 = self.dec2(torch.cat([self.up2(m), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        pooled = torch.cat([m.mean(dim=(2, 3)), m.amax(dim=(2, 3))], dim=1)
        if return_features:
            return self.out(d1), self.presence(pooled)[:, 0], m
        return self.out(d1), self.presence(pooled)[:, 0]


##### 4. Loss functions and the training function (used once per fold)

pixel_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(PIXEL_WEIGHT, device=device))
presence_loss_fn = nn.BCEWithLogitsLoss()

def dice_loss_fn(logits, target):
    prob = torch.sigmoid(logits)
    overlap = (prob * target).sum(dim=(1, 2, 3))
    total = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return 1 - ((2 * overlap + 1) / (total + 1)).mean()

def compute_loss(seg_logits, pres_logit, yb):
    frame_label = yb.amax(dim=(1, 2, 3))                   # 1 if the frame has any nerve pixel
    pixel = pixel_loss_fn(seg_logits, yb)
    dice = dice_loss_fn(seg_logits, yb)
    presence = presence_loss_fn(pres_logit, frame_label)
    return pixel + dice + HEAD_WEIGHT * presence, pixel, dice, presence

def train_model(train_ids, val_ids, seed, fold):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_set = TensorDataset(torch.from_numpy(X[train_ids]).unsqueeze(1),
                              torch.from_numpy(Y[train_ids]).unsqueeze(1))
    # nerve frames are drawn more often (step 2)
    frame_has_nerve = has_nerve[train_ids]
    weights = np.where(frame_has_nerve, (~frame_has_nerve).sum() / frame_has_nerve.sum(), 1.0)
    sampler = WeightedRandomSampler(torch.from_numpy(weights).double(),
                                    num_samples=len(train_ids), replacement=True)
    loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler)
    x_val = torch.from_numpy(X[val_ids]).unsqueeze(1)
    y_val = torch.from_numpy(Y[val_ids]).unsqueeze(1)

    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = []
    for epoch in range(EPOCHS):
        model.train()
        sums = np.zeros(4)                                  # total, pixel, dice, presence
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            seg_logits, pres_logit = model(xb)
            loss, pixel, dice, presence = compute_loss(seg_logits, pres_logit, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sums += np.array([loss.item(), pixel.item(), dice.item(), presence.item()]) * len(xb)
        sums /= len(train_ids)
        # loss on the held-out fold, only for the learning curve
        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for start in range(0, len(val_ids), 128):
                xb = x_val[start:start + 128].to(device)
                yb = y_val[start:start + 128].to(device)
                seg_logits, pres_logit = model(xb)
                val_total += compute_loss(seg_logits, pres_logit, yb)[0].item() * len(xb)
        val_loss = val_total / len(val_ids)
        history.append({"fold": fold, "epoch": epoch + 1, "train_loss": sums[0], "pixel_loss": sums[1],
                        "dice_loss": sums[2], "presence_loss": sums[3], "val_loss": val_loss})
        mlflow.log_metric(f"train_loss_fold{fold}", sums[0], step=epoch + 1)
        mlflow.log_metric(f"val_loss_fold{fold}", val_loss, step=epoch + 1)
    print(f"  fold {fold + 1}/{N_FOLDS}: trained on {len(train_ids)} frames, "
          f"last epoch train loss {history[-1]['train_loss']:.4f}, held-out loss {history[-1]['val_loss']:.4f}")
    return model, history


##### 5. Cross-validation loop: train on 4 groups, keep the 5th out

models, all_history = [], []
t0 = time.time()
for k, (tr, va) in enumerate(folds):
    model, history = train_model(dev_idx[tr], dev_idx[va], seed=SEED + k, fold=k)
    models.append(model)
    all_history += history
print(f"trained {N_FOLDS} models in {time.time() - t0:.0f} s")

history_df = pd.DataFrame(all_history)
history_df.to_csv(os.path.join(OUT_DIR, "tables", "train_history.csv"), index=False)


##### 6. Save the models
# state_dict files: used by steps 9 and 10 (Grad-CAM needs the normal PyTorch model).
# TorchScript files: for the C++ program (libtorch). Same export and check as step 7.

example = torch.rand(1, 1, H, W)
for k, model in enumerate(models):
    model.eval().cpu()
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "models", f"unet_fold{k}.pt"))
    ts_path = os.path.join(OUT_DIR, "models", f"nerve_unet_fold{k}_torchscript.pt")
    torch.jit.trace(model, example).save(ts_path)
    with torch.no_grad():
        seg_a, pres_a = model(example)
        seg_b, pres_b = torch.jit.load(ts_path)(example)
    difference = max((seg_a - seg_b).abs().max().item(), (pres_a - pres_b).abs().max().item())
    assert difference < 1e-4, f"exported model {k} differs from the Python model"
print(f"saved {N_FOLDS} state_dicts and {N_FOLDS} TorchScript models (reloaded outputs match to 1e-4)")

# Reference files for the C++ side: one real test frame and what model 1 gives for it (raw float32).
reference_input = torch.from_numpy(X[test_idx[:1]]).unsqueeze(1)
with torch.no_grad():
    reference_seg, reference_pres = models[0](reference_input)
reference_input.numpy().astype(np.float32).tofile(os.path.join(OUT_DIR, "models", "nerve_reference_input.bin"))
reference_seg.numpy().astype(np.float32).tofile(os.path.join(OUT_DIR, "models", "nerve_reference_seg_logits_fold0.bin"))
reference_pres.numpy().astype(np.float32).tofile(os.path.join(OUT_DIR, "models", "nerve_reference_presence_logit_fold0.bin"))


##### 7. Save the results in MLflow

mlflow.log_metric("train_seconds", time.time() - t0)
mlflow.log_artifacts(os.path.join(OUT_DIR, "models"), artifact_path="models")
mlflow.log_artifact(os.path.join(OUT_DIR, "split.npz"))
mlflow.log_artifact(os.path.join(OUT_DIR, "tables", "train_history.csv"))
try:
    mlflow.log_artifact(__file__)       # saves a copy of this code with the run
except NameError:
    pass                                # in a notebook there is no file to copy
mlflow.end_run()
print("saved run to MLflow:", RUN_NAME)
print(f"next: run step9_diagnostics.py  (reads {OUT_DIR})")
