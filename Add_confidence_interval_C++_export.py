##### Step 7: confidence intervals and export for C++
# Same pipeline as step 6. Changes are marked with ">>> STEP 7":
#   1. Confidence intervals: how sure are we about the test numbers? The test set has only a few
#      patients, so a gain could be luck. We redraw the test patients at random 2000 times and
#      recompute the score each time. The middle 95% of the results is the interval.
#      Patients are redrawn, not frames, because frames of one patient are not independent.
#   2. Verdicts: BETTER if the whole interval of a gain is above 0, WORSE if it is below 0,
#      INCONCLUSIVE if it crosses 0. INCONCLUSIVE does not mean "no effect".
#   3. Export: the 5 trained models are saved as TorchScript files that a C++ program
#      can load with libtorch. A JSON file lists every step around the model.
# The run is saved with MLflow, so the steps can be compared later.
# Needs:  pip install mlflow   (in a Kaggle notebook:  !pip install mlflow)
# Data: Kaggle "Ultrasound Nerve Segmentation" (neck ultrasound, brachial plexus).

import glob
import json
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
N_FOLDS = 5                         # from step 6
N_BOOT = 2000                       # >>> STEP 7: how many times the test patients are redrawn
PRES_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]    # values to try for the detection threshold
PIX_GRID = [0.3, 0.4, 0.5, 0.6]                          # values to try for the pixel threshold
AREA_GRID = [0, 25, 50, 100, 200, 300, 400]              # smallest allowed blob, in pixels on the 96x128 grid
RUN_NAME = "step7_ci_and_export"    # the name of this run in the MLflow table

torch.manual_seed(SEED)
np.random.seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"


##### 0. Start the MLflow run
# The history is kept in the file mlflow.db (plus a folder mlruns for saved code).
# Both are created in the folder you run from.

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nerve_segmentation")
mlflow.start_run(run_name=RUN_NAME)
mlflow.log_params({"step": 7, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "image_size": f"{H}x{W}",
                   "oversample": True, "pixel_weight": PIXEL_WEIGHT, "dice_loss": True,
                   "detection_head": True, "head_weight": HEAD_WEIGHT,
                   "tuned_thresholds": True, "largest_blob_only": True,
                   "flip_average": True, "n_folds": N_FOLDS, "ensemble_models": N_FOLDS,
                   "n_boot": N_BOOT, "exported_for_cpp": True})


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
# First the test patients are set aside (same as steps 1 to 6).
# The other patients ("dev") are split into 5 groups for cross-validation.

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
dev_idx, test_idx = next(splitter.split(X, groups=patients))
assert not (set(patients[dev_idx]) & set(patients[test_idx])), "a patient is in train and test"
print(f"dev: {len(dev_idx)} frames, {len(set(patients[dev_idx]))} patients   "
      f"test: {len(test_idx)} frames, {len(set(patients[test_idx]))} patients")

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


##### 4. Training and prediction, as functions (same as step 6)

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

def average_models(predictions):
    seg = np.mean([p[0] for p in predictions], axis=0, dtype=np.float32)
    pres = np.mean([p[1] for p in predictions], axis=0)
    return seg, pres


##### 5. Cross-validation loop (same as step 6)

oof_seg = np.zeros((len(dev_idx), H, W), dtype=np.float16)
oof_pres = np.zeros(len(dev_idx), dtype=np.float32)
test_plain, test_flip, models = [], [], []

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


##### 7. Choose the three thresholds on the out-of-fold predictions (same as step 6)

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


##### 8. Test results (same as step 6)

test_true = Y[test_idx] > 0.5
test_has = has_nerve[test_idx]
test_patients = patients[test_idx]
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


##### 9. Confidence intervals over the test patients
# >>> STEP 7: the bootstrap works on a list of per-frame values (for example the Dice of every test frame).
# For every test patient we keep four numbers: sum and count of nerve frames, sum and count of empty frames.
# Then we draw patients with replacement, add up their numbers, and compute the scores.
# Doing this N_BOOT times gives N_BOOT possible scores. The 2.5% and 97.5% marks are the 95% interval.

def patient_bootstrap(frame_values):
    rng = np.random.default_rng(SEED)
    unique_patients = np.unique(test_patients)
    per_patient = np.array([[frame_values[(test_patients == p) & test_has].sum(),
                             ((test_patients == p) & test_has).sum(),
                             frame_values[(test_patients == p) & ~test_has].sum(),
                             ((test_patients == p) & ~test_has).sum()] for p in unique_patients])
    drawn = rng.integers(0, len(unique_patients), (N_BOOT, len(unique_patients)))
    totals = per_patient[drawn].sum(axis=1)                         # shape (N_BOOT, 4)
    nerve = totals[:, 0] / np.maximum(totals[:, 1], 1)
    empty = totals[:, 2] / np.maximum(totals[:, 3], 1)
    balanced = 0.5 * (nerve + empty)
    return {"nerve_dice": (np.percentile(nerve, 2.5), np.percentile(nerve, 97.5)),
            "balanced": (np.percentile(balanced, 2.5), np.percentile(balanced, 97.5))}

# >>> STEP 7: three outcomes instead of two.
def verdict(low, high):
    if low > 0:
        return "BETTER"
    if high < 0:
        return "WORSE"
    return "INCONCLUSIVE"

def point_value(frame_values, kind):
    nerve, empty = frame_values[test_has].mean(), frame_values[~test_has].mean()
    return nerve if kind == "nerve_dice" else 0.5 * (nerve + empty)

# per-frame Dice of the final pipeline, of one single model without flip (fold 1), and of "always empty"
final_dice = frame_dice(make_masks(*average_models(test_flip), **POST), test_true)
single_dice = frame_dice(make_masks(*test_plain[0], **POST), test_true)
empty_dice = (~test_has).astype(float)

ci_final = patient_bootstrap(final_dice)
print("\nfinal pipeline (5 models, flip), 95% interval over test patients:")
for kind, (low, high) in ci_final.items():
    print(f"    {kind:11s} {point_value(final_dice, kind):.3f}   [{low:.3f}, {high:.3f}]")

# Paired comparisons: the same test frames, so we look at the per-frame difference.
for label, other_dice in [("always empty", empty_dice), ("one model without flip", single_dice)]:
    difference = final_dice - other_dice
    print(f"\nfinal pipeline minus {label}:")
    for kind, (low, high) in patient_bootstrap(difference).items():
        print(f"    {kind:11s} gain {point_value(difference, kind):+.3f}   [{low:+.3f}, {high:+.3f}]   {verdict(low, high)}")


##### 10. Export for a C++ program (libtorch)
# >>> STEP 7: torch.jit.trace runs the model once with an example input and records every operation.
# The result is a file that C++ can load without Python.
# The model alone does not give these results. The steps around it must be repeated exactly in C++.
# So the JSON file lists them all.

example = torch.rand(1, 1, H, W)
for k, model in enumerate(models):
    model.eval().cpu()
    path = f"nerve_unet_fold{k}_torchscript.pt"
    torch.jit.trace(model, example).save(path)
    with torch.no_grad():
        seg_a, pres_a = model(example)
        seg_b, pres_b = torch.jit.load(path)(example)
    difference = max((seg_a - seg_b).abs().max().item(), (pres_a - pres_b).abs().max().item())
    assert difference < 1e-4, f"exported model {k} differs from the Python model"
print(f"\nexported {len(models)} TorchScript models (reloaded outputs match to 1e-4)")

# Reference files for the C++ side: one real test frame and what model 1 gives for it (raw float32).
# A C++ program that loads the model should reproduce these numbers.
reference_input = torch.from_numpy(X[test_idx[:1]]).unsqueeze(1)
with torch.no_grad():
    reference_seg, reference_pres = models[0](reference_input)
reference_input.numpy().astype(np.float32).tofile("nerve_reference_input.bin")
reference_seg.numpy().astype(np.float32).tofile("nerve_reference_seg_logits_fold0.bin")
reference_pres.numpy().astype(np.float32).tofile("nerve_reference_presence_logit_fold0.bin")

config = {
    "models": [f"nerve_unet_fold{k}_torchscript.pt" for k in range(N_FOLDS)],
    "input_shape": [1, 1, H, W],
    "preprocessing": ["load the image as grayscale",
                      f"resize to {W}x{H} (width x height) with bilinear interpolation (PIL Image.BILINEAR)",
                      "divide by 255, so values are float32 between 0 and 1"],
    "model_outputs": [f"segmentation logits, shape (1, 1, {H}, {W})", "presence logit, shape (1)"],
    "postprocessing": ["apply sigmoid to both outputs",
                       "do the same for the mirrored input (flip left-right), flip the segmentation back, average the two",
                       "average over all 5 models",
                       f"if the presence probability is below {POST['pres_thr']}: empty mask",
                       f"set pixels with probability above {POST['pix_thr']} to nerve",
                       "keep only the largest connected blob",
                       f"if the blob has fewer than {POST['min_area']} pixels: empty mask"],
    "thresholds": best,
    "reference_files": {"input": "nerve_reference_input.bin",
                        "segmentation_logits": "nerve_reference_seg_logits_fold0.bin",
                        "presence_logit": "nerve_reference_presence_logit_fold0.bin",
                        "note": "raw float32, model 1 only, no flip"},
}
with open("nerve_unet_config.json", "w") as f:
    json.dump(config, f, indent=2)
print("saved nerve_unet_config.json and the reference files")
# This only checks the file against itself in Python. Run the reference input through libtorch
# in the C++ program and compare with the reference output before trusting the export.
#
# Sketch of the C++ side (needs libtorch, not tested here):
#   torch::jit::script::Module m = torch::jit::load("nerve_unet_fold0_torchscript.pt");
#   auto out = m.forward({input}).toTuple();          // input: float tensor (1, 1, 96, 128), values 0 to 1
#   at::Tensor seg_logits = out->elements()[0].toTensor();
#   at::Tensor pres_logit = out->elements()[1].toTensor();


##### 11. Save the results in MLflow
# Same metric names in every step, so the runs line up in one table.
# The saved numbers are the final pipeline: 5 models with flip averaging.

final = results["5 models, flip"]
mlflow.log_params({"tuned_pres_thr": best["pres_thr"], "tuned_pix_thr": best["pix_thr"],
                   "tuned_min_area": best["min_area"]})
for name, value in final.items():
    mlflow.log_metric(name, float(value))
mlflow.log_metric("always_empty_dice", float(always_empty))
mlflow.log_metric("single_model_balanced", float(results["one model, no flip"]["balanced"]))
mlflow.log_metric("balanced_ci_low", float(ci_final["balanced"][0]))
mlflow.log_metric("balanced_ci_high", float(ci_final["balanced"][1]))
mlflow.log_metric("nerve_dice_ci_low", float(ci_final["nerve_dice"][0]))
mlflow.log_metric("nerve_dice_ci_high", float(ci_final["nerve_dice"][1]))
for path in config["models"] + ["nerve_unet_config.json"] + list(config["reference_files"].values())[:3]:
    mlflow.log_artifact(path)           # the exported files are stored with the run
try:
    mlflow.log_artifact(__file__)       # saves a copy of this code with the run
except NameError:
    pass                                # in a notebook there is no file to copy
mlflow.end_run()
print("saved run to MLflow:", RUN_NAME)



######### output #####
'''
5635 frames, 47 patients, nerve in 41% of frames
dev: 4436 frames, 37 patients   test: 1199 frames, 10 patients
  fold 1/5: trained on 3596 frames, last epoch loss 1.0726
  fold 2/5: trained on 3596 frames, last epoch loss 0.9414
  fold 3/5: trained on 3596 frames, last epoch loss 0.9160
  fold 4/5: trained on 3478 frames, last epoch loss 0.9898
  fold 5/5: trained on 3478 frames, last epoch loss 1.0014
trained 5 models in 334 s
chosen pres_thr  = 0.5   (out-of-fold balanced score 0.556)
chosen pix_thr   = 0.4   (out-of-fold balanced score 0.574)
chosen min_area  = 200   (out-of-fold balanced score 0.593)

--- test results ---
always empty:        balanced 0.500   Dice all 0.494
one model, no flip   balanced 0.612   Dice all 0.609   nerve frames 0.370   found 52%   empty kept empty 85%
one model, flip      balanced 0.573   Dice all 0.568   nerve frames 0.229   found 32%   empty kept empty 92%
5 models, no flip    balanced 0.652   Dice all 0.649   nerve frames 0.430   found 57%   empty kept empty 87%
5 models, flip       balanced 0.594   Dice all 0.590   nerve frames 0.241   found 31%   empty kept empty 95%

final pipeline (5 models, flip), 95% interval over test patients:
    nerve_dice  0.241   [0.106, 0.362]
    balanced    0.594   [0.534, 0.650]

final pipeline minus always empty:
    nerve_dice  gain +0.241   [+0.106, +0.362]   BETTER
    balanced    gain +0.094   [+0.034, +0.150]   BETTER

final pipeline minus one model without flip:
    nerve_dice  gain -0.001   [-0.136, +0.125]   INCONCLUSIVE
    balanced    gain +0.081   [+0.003, +0.167]   BETTER

exported 5 TorchScript models (reloaded outputs match to 1e-4)
saved nerve_unet_config.json and the reference files
saved run to MLflow: step7_ci_and_export5635 frames, 47 patients, nerve in 41% of frames
dev: 4436 frames, 37 patients   test: 1199 frames, 10 patients
  fold 1/5: trained on 3596 frames, last epoch loss 1.0726
  fold 2/5: trained on 3596 frames, last epoch loss 0.9414
  fold 3/5: trained on 3596 frames, last epoch loss 0.9160
  fold 4/5: trained on 3478 frames, last epoch loss 0.9898
  fold 5/5: trained on 3478 frames, last epoch loss 1.0014
trained 5 models in 334 s
chosen pres_thr  = 0.5   (out-of-fold balanced score 0.556)
chosen pix_thr   = 0.4   (out-of-fold balanced score 0.574)
chosen min_area  = 200   (out-of-fold balanced score 0.593)

--- test results ---
always empty:        balanced 0.500   Dice all 0.494
one model, no flip   balanced 0.612   Dice all 0.609   nerve frames 0.370   found 52%   empty kept empty 85%
one model, flip      balanced 0.573   Dice all 0.568   nerve frames 0.229   found 32%   empty kept empty 92%
5 models, no flip    balanced 0.652   Dice all 0.649   nerve frames 0.430   found 57%   empty kept empty 87%
5 models, flip       balanced 0.594   Dice all 0.590   nerve frames 0.241   found 31%   empty kept empty 95%

final pipeline (5 models, flip), 95% interval over test patients:
    nerve_dice  0.241   [0.106, 0.362]
    balanced    0.594   [0.534, 0.650]

final pipeline minus always empty:
    nerve_dice  gain +0.241   [+0.106, +0.362]   BETTER
    balanced    gain +0.094   [+0.034, +0.150]   BETTER

final pipeline minus one model without flip:
    nerve_dice  gain -0.001   [-0.136, +0.125]   INCONCLUSIVE
    balanced    gain +0.081   [+0.003, +0.167]   BETTER

exported 5 TorchScript models (reloaded outputs match to 1e-4)
saved nerve_unet_config.json and the reference files
saved run to MLflow: step7_ci_and_export
'''
