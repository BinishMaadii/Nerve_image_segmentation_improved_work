
##### Step 6: flip averaging, cross-validation and an ensemble of 5 models
# Same pipeline as step 5. Changes are marked with ">>> STEP 6":
#   1. Cross-validation: the non-test patients are split into 5 groups. 5 models are trained,
#      and each one leaves one group out. So every one of these patients is predicted by a model
#      that never saw that patient ("out-of-fold" predictions).
#      These predictions replace the small validation group for choosing thresholds.
#   2. Flip averaging: each image is predicted twice (normal and mirrored left-right).
#      The mirrored prediction is flipped back, then both are averaged.
#   3. Ensemble: the 5 models all predict the test patients, and their outputs are averaged.
# The test patients are the same as in steps 1 to 5. They are used once, at the end.
# The results show each idea alone, so you can see what it adds.
# Training takes about 5 times longer than before, because 5 models are trained.
# The run is saved with MLflow, so the steps can be compared later.
# Needs:  pip install mlflow   (in a Kaggle notebook:  !pip install mlflow)
# Data: Kaggle "Ultrasound Nerve Segmentation" (neck ultrasound, brachial plexus).

import glob
import os
import re
import time

import mlflow
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy import ndimage
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

DATA_DIR = "/kaggle/input/competitions/ultrasound-nerve-segmentation"
H, W = 96, 128      # images are shrunk to this size
EPOCHS = 10
BATCH_SIZE = 32
SEED = 0
PIXEL_WEIGHT = 5.0                  # from step 3
HEAD_WEIGHT = 0.5                   # from step 4
N_FOLDS = 5                         # >>> STEP 6: number of groups = number of models
PRES_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]    # values to try for the detection threshold
PIX_GRID = [0.3, 0.4, 0.5, 0.6]                          # values to try for the pixel threshold
AREA_GRID = [0, 25, 50, 100, 200, 300, 400]              # smallest allowed blob, in pixels on the 96x128 grid
RUN_NAME = "step6_flip_cv_ensemble"  # the name of this run in the MLflow table

torch.manual_seed(SEED)
np.random.seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"


##### 0. Start the MLflow run
# The history is kept in the file mlflow.db (plus a folder mlruns for saved code).
# Both are created in the folder you run from.

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nerve_segmentation")
mlflow.start_run(run_name=RUN_NAME)
mlflow.log_params({"step": 6, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "image_size": f"{H}x{W}",
                   "oversample": True, "pixel_weight": PIXEL_WEIGHT, "dice_loss": True,
                   "detection_head": True, "head_weight": HEAD_WEIGHT,
                   "tuned_thresholds": True, "largest_blob_only": True,
                   "flip_average": True, "n_folds": N_FOLDS, "ensemble_models": N_FOLDS})


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
# First the test patients are set aside (same as steps 1 to 5).
# The other patients ("dev") are split into 5 groups for cross-validation.

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
dev_idx, test_idx = next(splitter.split(X, groups=patients))
assert not (set(patients[dev_idx]) & set(patients[test_idx])), "a patient is in train and test"
print(f"dev: {len(dev_idx)} frames, {len(set(patients[dev_idx]))} patients   "
      f"test: {len(test_idx)} frames, {len(set(patients[test_idx]))} patients")

# >>> STEP 6: GroupKFold keeps all frames of a patient in the same group.
folds = list(GroupKFold(n_splits=N_FOLDS).split(dev_idx, groups=patients[dev_idx]))


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


##### 4. Training and prediction, as functions (they are used once per fold)

pixel_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(PIXEL_WEIGHT, device=device))
presence_loss_fn = nn.BCEWithLogitsLoss()

def dice_loss_fn(logits, target):
    prob = torch.sigmoid(logits)
    overlap = (prob * target).sum(dim=(1, 2, 3))
    total = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return 1 - ((2 * overlap + 1) / (total + 1)).mean()

def train_model(train_ids, seed, fold):
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

    model = UNet().to(device)
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
        mlflow.log_metric(f"train_loss_fold{fold}", total / len(train_ids), step=epoch + 1)
    print(f"  fold {fold + 1}/{N_FOLDS}: trained on {len(train_ids)} frames, last epoch loss {total / len(train_ids):.4f}")
    return model

# >>> STEP 6: with flip_average=True each frame is predicted twice, normal and mirrored left-right.
# The mirrored segmentation is flipped back before averaging.
def predict(model, ids, flip_average):
    model.eval()
    seg, pres = [], []
    with torch.no_grad():
        for start in range(0, len(ids), 64):
            xb = torch.from_numpy(X[ids[start:start + 64]]).unsqueeze(1).to(device)
            seg_logits, pres_logit = model(xb)
            seg_prob, pres_prob = torch.sigmoid(seg_logits), torch.sigmoid(pres_logit)
            if flip_average:
                seg_logits2, pres_logit2 = model(xb.flip(-1))
                seg_prob = (seg_prob + torch.sigmoid(seg_logits2).flip(-1)) / 2
                pres_prob = (pres_prob + torch.sigmoid(pres_logit2)) / 2
            seg.append(seg_prob[:, 0].cpu().numpy().astype(np.float16))     # float16 saves memory
            pres.append(pres_prob.cpu().numpy())
    return np.concatenate(seg), np.concatenate(pres)

# >>> STEP 6: average the outputs of several models.
def average_models(predictions):
    seg = np.mean([p[0] for p in predictions], axis=0, dtype=np.float32)
    pres = np.mean([p[1] for p in predictions], axis=0)
    return seg, pres


##### 5. Cross-validation loop
# Each model is trained on 4 groups and predicts the 5th group (out-of-fold) and the test patients.

oof_seg = np.zeros((len(dev_idx), H, W), dtype=np.float16)    # out-of-fold probabilities for every dev frame
oof_pres = np.zeros(len(dev_idx), dtype=np.float32)
test_plain, test_flip, models = [], [], []                    # test predictions of each model

t0 = time.time()
for k, (tr, va) in enumerate(folds):
    model = train_model(dev_idx[tr], seed=SEED + k, fold=k)
    oof_seg[va], oof_pres[va] = predict(model, dev_idx[va], flip_average=True)
    test_plain.append(predict(model, test_idx, flip_average=False))
    test_flip.append(predict(model, test_idx, flip_average=True))
    models.append(model)
print(f"trained {N_FOLDS} models in {time.time() - t0:.0f} s")


##### 6. Clean-up and scoring functions (same as step 5)

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

def frame_dice(pred_masks, true_masks):
    overlap = (pred_masks & true_masks).sum(axis=(1, 2))
    total = pred_masks.sum(axis=(1, 2)) + true_masks.sum(axis=(1, 2))
    return np.where(total == 0, 1.0, 2 * overlap / np.maximum(total, 1))

def balanced_score(dice, frame_has_nerve):
    return 0.5 * dice[frame_has_nerve].mean() + 0.5 * dice[~frame_has_nerve].mean()


##### 7. Choose the three thresholds on the out-of-fold predictions
# >>> STEP 6: now ALL dev patients are used for this, not a small validation group.
# Every one of them was predicted by a model that never saw them, so the choice is fair.

oof_true = Y[dev_idx] > 0.5
oof_has = has_nerve[dev_idx]

def oof_score(pres_thr, pix_thr, min_area):
    oof_masks = make_masks(oof_seg, oof_pres, pres_thr, pix_thr, min_area, largest_only=True)
    return balanced_score(frame_dice(oof_masks, oof_true), oof_has)

best = {"pres_thr": 0.5, "pix_thr": 0.5, "min_area": 0}
for name, grid in [("pres_thr", PRES_GRID), ("pix_thr", PIX_GRID), ("min_area", AREA_GRID)]:
    scores = []
    for value in grid:
        trial = {**best, name: value}
        scores.append(oof_score(**trial))
    best[name] = grid[int(np.argmax(scores))]
    print(f"chosen {name:9s} = {best[name]}   (out-of-fold balanced score {max(scores):.3f})")
POST = {**best, "largest_only": True}


##### 8. Test results
# The test patients are used once. Four versions are compared, with the same thresholds:
#   one model without flip, one model with flip, 5 models without flip, 5 models with flip.
# "One model" is the average over the 5 single models.

test_true = Y[test_idx] > 0.5
test_has = has_nerve[test_idx]
always_empty = (~test_has).mean()

def report(seg_probs, pres_probs):
    pred_masks = make_masks(seg_probs, pres_probs, **POST)
    d = frame_dice(pred_masks, test_true)
    found = pred_masks.any(axis=(1, 2))
    return {"dice_all": d.mean(),
            "dice_nerve_frames": d[test_has].mean(),
            "nerve_found": found[test_has].mean(),
            "empty_kept_empty": (~found[~test_has]).mean(),
            "balanced": balanced_score(d, test_has)}

def mean_of_reports(reports):
    return {key: np.mean([r[key] for r in reports]) for key in reports[0]}

results = {
    "one model, no flip":  mean_of_reports([report(s, p) for s, p in test_plain]),
    "one model, flip":     mean_of_reports([report(s, p) for s, p in test_flip]),
    "5 models, no flip":   report(*average_models(test_plain)),
    "5 models, flip":      report(*average_models(test_flip)),
}

print("\n--- test results ---")
print(f"always empty:        balanced 0.500   Dice all {always_empty:.3f}")
for name, m in results.items():
    print(f"{name:19s}  balanced {m['balanced']:.3f}   Dice all {m['dice_all']:.3f}   nerve frames {m['dice_nerve_frames']:.3f}   "
          f"found {m['nerve_found']:.0%}   empty kept empty {m['empty_kept_empty']:.0%}")


##### 9. Save the results in MLflow
# Same metric names in every step, so the runs line up in one table.
# The saved numbers are the final pipeline: 5 models with flip averaging.

final = results["5 models, flip"]
mlflow.log_params({"tuned_pres_thr": best["pres_thr"], "tuned_pix_thr": best["pix_thr"],
                   "tuned_min_area": best["min_area"]})
for name, value in final.items():
    mlflow.log_metric(name, float(value))
mlflow.log_metric("always_empty_dice", float(always_empty))
mlflow.log_metric("single_model_balanced", float(results["one model, no flip"]["balanced"]))
try:
    mlflow.log_artifact(__file__)       # saves a copy of this code with the run
except NameError:
    pass                                # in a notebook there is no file to copy
mlflow.end_run()
print("saved run to MLflow:", RUN_NAME)



########## ouputs#####
'''


##### Step 6: flip averaging, cross-validation and an ensemble of 5 models

##### 8. Test results
# The test patients are used once. Four versions are compared, with the same thresholds:
#   one model without flip, one model with flip, 5 models without flip, 5 models with flip.
# "One model" is the average over the 5 single models.

test_true = Y[test_idx] > 0.5
test_has = has_nerve[test_idx]
always_empty = (~test_has).mean()

def report(seg_probs, pres_probs):
    pred_masks = make_masks(seg_probs, pres_probs, **POST)
    d = frame_dice(pred_masks, test_true)
    found = pred_masks.any(axis=(1, 2))
    return {"dice_all": d.mean(),
            "dice_nerve_frames": d[test_has].mean(),
            "nerve_found": found[test_has].mean(),
            "empty_kept_empty": (~found[~test_has]).mean(),
            "balanced": balanced_score(d, test_has)}

def mean_of_reports(reports):
    return {key: np.mean([r[key] for r in reports]) for key in reports[0]}

results = {
    "one model, no flip":  mean_of_reports([report(s, p) for s, p in test_plain]),
    "one model, flip":     mean_of_reports([report(s, p) for s, p in test_flip]),
    "5 models, no flip":   report(*average_models(test_plain)),
    "5 models, flip":      report(*average_models(test_flip)),
}

print("\n--- test results ---")
print(f"always empty:        balanced 0.500   Dice all {always_empty:.3f}")
for name, m in results.items():
    print(f"{name:19s}  balanced {m['balanced']:.3f}   Dice all {m['dice_all']:.3f}   nerve frames {m['dice_nerve_frames']:.3f}   "
          f"found {m['nerve_found']:.0%}   empty kept empty {m['empty_kept_empty']:.0%}")


##### 9. Save the results in MLflow
# Same metric names in every step, so the runs line up in one table.
# The saved numbers are the final pipeline: 5 models with flip averaging.

final = results["5 models, flip"]
mlflow.log_params({"tuned_pres_thr": best["pres_thr"], "tuned_pix_thr": best["pix_thr"],
                   "tuned_min_area": best["min_area"]})
for name, value in final.items():
    mlflow.log_metric(name, float(value))
mlflow.log_metric("always_empty_dice", float(always_empty))
mlflow.log_metric("single_model_balanced", float(results["one model, no flip"]["balanced"]))
try:
    mlflow.log_artifact(__file__)       # saves a copy of this code with the run
except NameError:
    pass                                # in a notebook there is no file to copy
mlflow.end_run()
print("saved run to MLflow:", RUN_NAME)
5635 frames, 47 patients, nerve in 41% of frames
dev: 4436 frames, 37 patients   test: 1199 frames, 10 patients
  fold 1/5: trained on 3596 frames, last epoch loss 1.0656
  fold 2/5: trained on 3596 frames, last epoch loss 1.0367
  fold 3/5: trained on 3596 frames, last epoch loss 0.9197
  fold 4/5: trained on 3478 frames, last epoch loss 1.0763
  fold 5/5: trained on 3478 frames, last epoch loss 1.0422
trained 5 models in 332 s
chosen pres_thr  = 0.8   (out-of-fold balanced score 0.500)
chosen pix_thr   = 0.4   (out-of-fold balanced score 0.502)
chosen min_area  = 0   (out-of-fold balanced score 0.502)

--- test results ---
always empty:        balanced 0.500   Dice all 0.494
one model, no flip   balanced 0.516   Dice all 0.510   nerve frames 0.044   found 6%   empty kept empty 99%
one model, flip      balanced 0.502   Dice all 0.495   nerve frames 0.003   found 1%   empty kept empty 100%
5 models, no flip    balanced 0.502   Dice all 0.496   nerve frames 0.004   found 0%   empty kept empty 100%
5 models, flip       balanced 0.500   Dice all 0.494   nerve frames 0.000   found 0%   empty kept empty 100%
saved run to MLflow: step6_flip_cv_ensemble

'''
