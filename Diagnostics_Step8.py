##### Step 9: diagnostics (tables and plots)
# Reads the 5 models and the split saved by step8_train.py. Trains nothing.
# It answers four questions:
#   A. Which way of using the models is best: flip averaging on/off, default or tuned thresholds,
#      one model or 5 models? (step 7 found that flip + 5 models finds almost no nerve)
#   B. How good is the detection ("is there a nerve?") on its own? ROC, PR, calibration.
#   C. How good is the segmentation when there is a nerve? Dice, IoU, precision, recall, size effect.
#   D. How much do the numbers move between folds and between patients?
# Every number comes with a 95% interval. The intervals redraw whole PATIENTS (not frames),
# because frames of one patient are not independent.
#
# Rule used here: nothing is chosen on the test patients. The configuration is chosen on the
# out-of-fold (OOF) predictions of the 37 non-test patients. The test patients are only reported.
# "tuned (cross-fitted)" means: thresholds are tuned on 4 folds and applied to the 5th, so the
# score is not inflated by tuning and scoring on the same frames.
# Simple options win ties: tuned thresholds or flip averaging are only chosen if they beat the
# simplest option (no flip, default thresholds) by at least 0.005 balanced score out-of-fold.
#
# Output (in OUT_DIR): tables/*.csv, tables/report.md, plots/*.png, final_config.json,
#                      models/nerve_unet_config.json (for the C++ side)
# Time: a few minutes (the threshold search is the slow part).
# Needs:  pip install mlflow torch scikit-learn scipy pandas matplotlib

import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import ndimage
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

OUT_DIR = os.environ.get("OUT_DIR", "outputs")
H, W = 96, 128
SEED = 0
N_BOOT = int(os.environ.get("N_BOOT", 2000))          # patient redraws for most intervals
N_BOOT_AUC = int(os.environ.get("N_BOOT_AUC", 500))   # for AUROC / AP (slower per redraw)
PRES_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]    # values to try for the detection threshold
PIX_GRID = [0.3, 0.4, 0.5, 0.6]                          # values to try for the pixel threshold
AREA_GRID = [0, 25, 50, 100, 200, 300, 400]              # smallest allowed blob, in pixels on the 96x128 grid
RUN_NAME = "step9_diagnostics"

TABLE_DIR = os.path.join(OUT_DIR, "tables")
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
rng = np.random.default_rng(SEED)

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 130, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
C_NERVE, C_EMPTY, C_GREY, C_RED = "#1b6ca8", "#d9822b", "#8a8a8a", "#c0392b"

mlflow.set_tracking_uri(f"sqlite:///{os.path.join(OUT_DIR, 'mlflow.db')}")
mlflow.set_experiment("nerve_segmentation")
mlflow.start_run(run_name=RUN_NAME)
mlflow.log_params({"step": 9, "n_boot": N_BOOT, "n_boot_auc": N_BOOT_AUC})


##### 1. Load the cache, the split and the 5 models

cache = np.load(os.path.join(OUT_DIR, "data_cache.npz"))
X = cache["X"].astype(np.float32) / 255.0
Y = cache["Y"].astype(np.float32)
patients = cache["patients"]
has_nerve = Y.sum(axis=(1, 2)) > 0
split = np.load(os.path.join(OUT_DIR, "split.npz"))
dev_idx, test_idx, fold_of_dev = split["dev_idx"], split["test_idx"], split["fold_of_dev"]
N_FOLDS = int(fold_of_dev.max()) + 1

oof_true, oof_has, oof_pat = Y[dev_idx] > 0.5, has_nerve[dev_idx], patients[dev_idx]
test_true, test_has, test_pat = Y[test_idx] > 0.5, has_nerve[test_idx], patients[test_idx]
print(f"dev: {len(dev_idx)} frames, {len(set(oof_pat))} patients   test: {len(test_idx)} frames, {len(set(test_pat))} patients")

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
        self.presence = nn.Linear(64 * 2, 1)

    def forward(self, x, return_features=False):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        m = self.middle(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(m), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        pooled = torch.cat([m.mean(dim=(2, 3)), m.amax(dim=(2, 3))], dim=1)
        if return_features:
            return self.out(d1), self.presence(pooled)[:, 0], m
        return self.out(d1), self.presence(pooled)[:, 0]

models = []
for k in range(N_FOLDS):
    model = UNet()
    model.load_state_dict(torch.load(os.path.join(OUT_DIR, "models", f"unet_fold{k}.pt"), map_location="cpu"))
    models.append(model.to(device).eval())


##### 2. Predict: out-of-fold for the dev patients, and all 5 models for the test patients
# Both modes are kept: "no flip" and "flip" (the input and its mirror image are predicted and averaged).

def predict(model, ids, flip_average):
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

def predict_pair(model, ids):
    """Mask of the normal input and mask of the mirrored input (flipped back), both at 0.5."""
    a, b = [], []
    with torch.no_grad():
        for start in range(0, len(ids), 64):
            xb = torch.from_numpy(X[ids[start:start + 64]]).unsqueeze(1).to(device)
            a.append((torch.sigmoid(model(xb)[0])[:, 0] > 0.5).cpu().numpy())
            b.append((torch.sigmoid(model(xb.flip(-1))[0]).flip(-1)[:, 0] > 0.5).cpu().numpy())
    return np.concatenate(a), np.concatenate(b)

MODES = ["no flip", "flip"]
oof_seg = {m: np.zeros((len(dev_idx), H, W), dtype=np.float16) for m in MODES}
oof_pres = {m: np.zeros(len(dev_idx), dtype=np.float32) for m in MODES}
test_preds = {m: [] for m in MODES}                 # one (segmentation, presence) pair per model
pair_dice = []                                      # flip consistency per model
t0 = time.time()
for k, model in enumerate(models):
    val = np.where(fold_of_dev == k)[0]
    for mode in MODES:
        s, p = predict(model, dev_idx[val], flip_average=(mode == "flip"))
        oof_seg[mode][val], oof_pres[mode][val] = s, p
        test_preds[mode].append(predict(model, test_idx, flip_average=(mode == "flip")))
    a, b = predict_pair(model, test_idx[test_has])
    inter = (a & b).sum(axis=(1, 2))
    total = a.sum(axis=(1, 2)) + b.sum(axis=(1, 2))
    pair_dice.append(np.where(total == 0, np.nan, 2 * inter / np.maximum(total, 1)))
    print(f"  model {k + 1}/{N_FOLDS} predicted")
pair_dice = np.nanmean(np.stack(pair_dice), axis=0)       # per test nerve frame, mean over models
print(f"predictions took {time.time() - t0:.0f} s")


##### 3. Masks, scores and the patient bootstrap

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

def frame_dice(pred, true):
    overlap = (pred & true).sum(axis=(1, 2))
    total = pred.sum(axis=(1, 2)) + true.sum(axis=(1, 2))
    return np.where(total == 0, 1.0, 2 * overlap / np.maximum(total, 1))

def frame_overlap(pred, true):
    """Dice, IoU, precision, recall per frame (meant for frames that contain a nerve)."""
    inter = (pred & true).sum(axis=(1, 2)).astype(np.float64)
    p, t = pred.sum(axis=(1, 2)), true.sum(axis=(1, 2))
    return {"dice": np.where(p + t == 0, 1.0, 2 * inter / np.maximum(p + t, 1)),
            "iou": np.where(p + t == 0, 1.0, inter / np.maximum(p + t - inter, 1)),
            "precision": np.where(p == 0, 0.0, inter / np.maximum(p, 1)),
            "recall": np.where(t == 0, 0.0, inter / np.maximum(t, 1))}

def balanced_score(dice, frame_has_nerve):
    return 0.5 * dice[frame_has_nerve].mean() + 0.5 * dice[~frame_has_nerve].mean()

def average_models(predictions):
    seg = np.mean([p[0] for p in predictions], axis=0, dtype=np.float32)
    pres = np.mean([p[1] for p in predictions], axis=0)
    return seg, pres

def patient_bootstrap(stat_fns, groups, n_boot):
    """Redraw whole patients with replacement. stat_fns: {name: function(frame_index_array) -> number}.
    Returns {name: (low, high)}, the middle 95% of the redrawn values."""
    draw_rng = np.random.default_rng(SEED)
    ids = np.unique(groups)
    members = [np.where(groups == g)[0] for g in ids]
    values = {name: [] for name in stat_fns}
    for _ in range(n_boot):
        drawn = draw_rng.integers(0, len(ids), len(ids))
        idx = np.concatenate([members[j] for j in drawn])
        for name, fn in stat_fns.items():
            values[name].append(fn(idx))
    out = {}
    for name, v in values.items():
        v = np.array(v, dtype=float)
        out[name] = (np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)) if np.isfinite(v).any() else (np.nan, np.nan)
    return out

def evaluate(mask_list, true, has, groups):
    """Detection+segmentation score of one configuration.
    mask_list holds one mask array per model (one entry for out-of-fold rows). Results are averaged over the list."""
    d_list = [frame_dice(m, true) for m in mask_list]
    f_list = [m.any(axis=(1, 2)) for m in mask_list]

    def nerve_dice(idx):
        h = has[idx]
        return np.nan if not h.any() else float(np.mean([d[idx][h].mean() for d in d_list]))
    def found(idx):
        h = has[idx]
        return np.nan if not h.any() else float(np.mean([f[idx][h].mean() for f in f_list]))
    def empty_kept(idx):
        h = has[idx]
        return np.nan if h.all() else float(np.mean([(~f[idx][~h]).mean() for f in f_list]))
    def balanced(idx):
        return 0.5 * nerve_dice(idx) + 0.5 * empty_kept(idx)

    fns = {"balanced": balanced, "nerve_dice": nerve_dice, "found": found, "empty_kept": empty_kept}
    everything = np.arange(len(has))
    ci = patient_bootstrap(fns, groups, N_BOOT)
    row = {}
    for name, fn in fns.items():
        row[name], row[name + "_lo"], row[name + "_hi"] = fn(everything), ci[name][0], ci[name][1]
    return row

def fmt(value, lo, hi):
    return f"{value:.3f} [{lo:.3f}, {hi:.3f}]"

def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)

report_parts = []          # (title, note, dataframe) for report.md
def add_table(name, title, note, df_numeric, df_display=None):
    df_numeric.to_csv(os.path.join(TABLE_DIR, name + ".csv"), index=False)
    shown = df_display if df_display is not None else df_numeric
    report_parts.append((title, note, shown))
    print(f"\n=== {title} ===")
    if note:
        print(note)
    print(shown.to_string(index=False))


##### 4. Thresholds: default, tuned on all OOF frames, and tuned cross-fitted
# The search is the one of steps 5 to 7: one threshold at a time, best balanced score wins.

DEFAULT = {"pres_thr": 0.5, "pix_thr": 0.5, "min_area": 0}

def tune(seg, pres, true, has):
    best = dict(DEFAULT)
    for name, grid in [("pres_thr", PRES_GRID), ("pix_thr", PIX_GRID), ("min_area", AREA_GRID)]:
        scores = []
        for value in grid:
            trial = {**best, name: value}
            masks = make_masks(seg, pres, trial["pres_thr"], trial["pix_thr"], trial["min_area"], True)
            scores.append(balanced_score(frame_dice(masks, true), has))
        best[name] = grid[int(np.argmax(scores))]
    return best

tuned, cross_fit_masks = {}, {}
t0 = time.time()
for mode in MODES:
    tuned[mode] = tune(oof_seg[mode], oof_pres[mode], oof_true, oof_has)
    cf = np.zeros((len(dev_idx), H, W), dtype=bool)
    for k in range(N_FOLDS):
        fit, apply_to = fold_of_dev != k, fold_of_dev == k
        t = tune(oof_seg[mode][fit], oof_pres[mode][fit], oof_true[fit], oof_has[fit])
        cf[apply_to] = make_masks(oof_seg[mode][apply_to], oof_pres[mode][apply_to], t["pres_thr"], t["pix_thr"], t["min_area"], True)
    cross_fit_masks[mode] = cf
    print(f"tuned thresholds ({mode}): {tuned[mode]}")
print(f"threshold search took {time.time() - t0:.0f} s")


##### 5. Table A: the configuration grid

rows = []
def add_row(dataset, models_used, mode, settings_name, res):
    rows.append({"dataset": dataset, "models": models_used, "flip": mode, "settings": settings_name, **res})

empty_oof = [np.zeros_like(oof_true)]
empty_test = [np.zeros_like(test_true)]
add_row("OOF (37 pt)", "-", "-", "always empty", evaluate(empty_oof, oof_true, oof_has, oof_pat))
for mode in MODES:
    seg, pres = oof_seg[mode], oof_pres[mode]
    add_row("OOF (37 pt)", "1 (out-of-fold)", mode, "default",
            evaluate([make_masks(seg, pres, 0.5, 0.5, 0, False)], oof_true, oof_has, oof_pat))
    add_row("OOF (37 pt)", "1 (out-of-fold)", mode, "tuned, same frames (optimistic)",
            evaluate([make_masks(seg, pres, tuned[mode]["pres_thr"], tuned[mode]["pix_thr"], tuned[mode]["min_area"], True)],
                     oof_true, oof_has, oof_pat))
    add_row("OOF (37 pt)", "1 (out-of-fold)", mode, "tuned, cross-fitted",
            evaluate([cross_fit_masks[mode]], oof_true, oof_has, oof_pat))

add_row("test (10 pt)", "-", "-", "always empty", evaluate(empty_test, test_true, test_has, test_pat))
for mode in MODES:
    for settings_name, t, largest in [("default", DEFAULT, False), ("tuned", tuned[mode], True)]:
        single = [make_masks(s, p, t["pres_thr"], t["pix_thr"], t["min_area"], largest) for s, p in test_preds[mode]]
        add_row("test (10 pt)", f"1 (mean of {N_FOLDS})", mode, settings_name, evaluate(single, test_true, test_has, test_pat))
        ens_seg, ens_pres = average_models(test_preds[mode])
        add_row("test (10 pt)", f"{N_FOLDS} averaged", mode, settings_name,
                evaluate([make_masks(ens_seg, ens_pres, t["pres_thr"], t["pix_thr"], t["min_area"], largest)],
                         test_true, test_has, test_pat))

config_df = pd.DataFrame(rows)
display = config_df[["dataset", "models", "flip", "settings"]].copy()
for name, title in [("balanced", "balanced score"), ("nerve_dice", "nerve-frame Dice"),
                    ("found", "nerve found"), ("empty_kept", "empty kept empty")]:
    display[title] = [fmt(r[name], r[name + "_lo"], r[name + "_hi"]) for _, r in config_df.iterrows()]
add_table("A_configurations", "Table A: configurations (value [95% interval over patients])",
          "balanced score = half the Dice on nerve frames + half the Dice on empty frames. 'Always empty' scores 0.5.\n"
          "OOF rows: every frame is predicted by a model that never saw its patient.\n"
          "Test rows: the test patients were not used for training or for choosing anything.", config_df, display)

# Choose the final configuration on OOF only, using the honest value (cross-fitted for tuned).
candidates = {}
for mode in MODES:
    d = config_df[(config_df.dataset == "OOF (37 pt)") & (config_df.flip == mode)]
    candidates[(mode, "default")] = float(d[d.settings == "default"].balanced.iloc[0])
    candidates[(mode, "tuned")] = float(d[d.settings == "tuned, cross-fitted"].balanced.iloc[0])
# Simple options come first. A more complex option (tuned thresholds, flip averaging) must win by
# at least MARGIN, otherwise a difference of 0.001 would decide the configuration.
MARGIN = 0.005
final_mode, final_kind = ("no flip", "default")
for option in [("no flip", "tuned"), ("flip", "default"), ("flip", "tuned")]:
    if candidates[option] >= candidates[(final_mode, final_kind)] + MARGIN:
        final_mode, final_kind = option
final_settings = dict(DEFAULT) if final_kind == "default" else dict(tuned[final_mode])
final_largest = final_kind == "tuned"
print(f"\nchosen on OOF only: flip = {final_mode}, thresholds = {final_kind} {final_settings} "
      f"(OOF balanced {candidates[(final_mode, final_kind)]:.3f})")
final_config = {"flip_average": final_mode == "flip", "thresholds": final_kind, "settings": final_settings,
                "largest_blob_only": final_largest, "candidates_oof_balanced": {f"{m} / {s}": v for (m, s), v in candidates.items()},
                "ensemble": f"average of the {N_FOLDS} fold models (design choice, see Table A for its test result)"}
with open(os.path.join(OUT_DIR, "final_config.json"), "w") as f:
    json.dump(final_config, f, indent=2)

# the final pipeline on the test patients
fin_seg, fin_pres = average_models(test_preds[final_mode])
fin_masks = make_masks(fin_seg, fin_pres, final_settings["pres_thr"], final_settings["pix_thr"],
                       final_settings["min_area"], final_largest)
fin_dice = frame_dice(fin_masks, test_true)
fin_found = fin_masks.any(axis=(1, 2))
oof_final_masks = make_masks(oof_seg[final_mode], oof_pres[final_mode], final_settings["pres_thr"],
                             final_settings["pix_thr"], final_settings["min_area"], final_largest)


##### 6. Table B: detection ("is there a nerve in this frame?")

def ece(prob, label, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(prob, edges) - 1, 0, bins - 1)
    return float(sum((which == b).mean() * abs(prob[which == b].mean() - label[which == b].mean())
                     for b in range(bins) if (which == b).any()))

def det_row(name, score_list, has, groups, thr_list):
    """score_list: one score array per model (or one). Values are averaged over the list."""
    def auc(idx):
        h = has[idx]
        return np.nan if h.all() or not h.any() else float(np.mean([roc_auc_score(h, s[idx]) for s in score_list]))
    def ap(idx):
        h = has[idx]
        return np.nan if not h.any() else float(np.mean([average_precision_score(h, s[idx]) for s in score_list]))
    everything = np.arange(len(has))
    ci = patient_bootstrap({"auroc": auc, "ap": ap}, groups, N_BOOT_AUC)
    out = {"scores": name, "AUROC": fmt(auc(everything), *ci["auroc"]), "AP": fmt(ap(everything), *ci["ap"]),
           "prevalence": f"{has.mean():.2f}"}
    for thr in thr_list:
        sens = np.mean([(s[has] >= thr).mean() for s in score_list])
        spec = np.mean([(s[~has] < thr).mean() for s in score_list])
        out[f"sens/spec @{thr:.2f}"] = f"{sens:.2f} / {spec:.2f}"
    return out

def peak(seg):                                          # highest pixel probability per frame
    return seg.reshape(len(seg), -1).max(axis=1).astype(np.float32)

det_rows = []
thr_pair = [0.5, final_settings["pres_thr"]] if final_settings["pres_thr"] != 0.5 else [0.5]
for mode in MODES:
    det_rows.append(det_row(f"OOF, detection head, {mode}", [oof_pres[mode]], oof_has, oof_pat, thr_pair))
    det_rows.append(det_row(f"OOF, max pixel probability, {mode}", [peak(oof_seg[mode])], oof_has, oof_pat, [0.5]))
for mode in MODES:
    ens_seg, ens_pres = average_models(test_preds[mode])
    det_rows.append(det_row(f"test, head, {N_FOLDS} averaged, {mode}", [ens_pres], test_has, test_pat, thr_pair))
    det_rows.append(det_row(f"test, head, 1 model (mean), {mode}", [p for _, p in test_preds[mode]], test_has, test_pat, thr_pair))
    det_rows.append(det_row(f"test, max pixel prob., {N_FOLDS} averaged, {mode}", [peak(ens_seg)], test_has, test_pat, [0.5]))
det_df = pd.DataFrame(det_rows).fillna("-")
add_table("B_detection", "Table B: frame-level detection of a nerve",
          "AUROC and AP do not depend on a threshold. 'max pixel probability' uses the segmentation output instead of the head.\n"
          "If the head is not clearly better than the max pixel probability, the extra head adds little.", det_df)

# calibration of the head (final mode)
ens_pres_final = fin_pres
cal_rows = []
for name, prob, label in [("OOF", oof_pres[final_mode], oof_has), (f"test ({N_FOLDS} averaged)", ens_pres_final, test_has)]:
    cal_rows.append({"data": name, "ECE (10 bins)": f"{ece(prob, label.astype(float)):.3f}",
                     "mean prob. on nerve frames": f"{prob[label].mean():.2f}",
                     "mean prob. on empty frames": f"{prob[~label].mean():.2f}"})
add_table("B2_calibration", f"Table B2: calibration of the detection head ({final_mode})",
          "ECE = average gap between predicted probability and the real share of nerve frames (0 is perfect).",
          pd.DataFrame(cal_rows))


##### 7. Table C: segmentation quality on frames that contain a nerve
# Detection gating and clean-up are switched off here (pixel threshold 0.5), so this is the segmentation alone.

def seg_row(name, mask_list, true, has, groups):
    nerve = np.where(has)[0]
    per_model = [frame_overlap(m[nerve], true[nerve]) for m in mask_list]
    grp = groups[nerve]
    fns = {key: (lambda idx, key=key: float(np.mean([pm[key][idx].mean() for pm in per_model]))) for key in per_model[0]}
    ci = patient_bootstrap(fns, grp, N_BOOT)
    out = {"predictions": name}
    for key, fn in fns.items():
        out[key] = fmt(fn(np.arange(len(nerve))), *ci[key])
    return out

seg_rows = []
for mode in MODES:
    seg_rows.append(seg_row(f"OOF, 1 model, {mode}", [oof_seg[mode] > 0.5], oof_true, oof_has, oof_pat))
for mode in MODES:
    seg_rows.append(seg_row(f"test, 1 model (mean), {mode}", [s > 0.5 for s, _ in test_preds[mode]], test_true, test_has, test_pat))
    seg_rows.append(seg_row(f"test, {N_FOLDS} averaged, {mode}", [average_models(test_preds[mode])[0] > 0.5], test_true, test_has, test_pat))
add_table("C_segmentation", "Table C: segmentation quality on nerve frames only (no gating, pixel threshold 0.5)",
          "Dice/IoU measure the overlap. Precision = share of predicted pixels that are nerve. Recall = share of nerve pixels found.",
          pd.DataFrame(seg_rows))


##### 8. Table D: folds and patients

fold_rows = []
for k in range(N_FOLDS):
    sel = np.where(fold_of_dev == k)[0]
    d = frame_dice(oof_final_masks[sel], oof_true[sel])
    h = oof_has[sel]
    found = oof_final_masks[sel].any(axis=(1, 2))
    auc = roc_auc_score(h, oof_pres[final_mode][sel]) if h.any() and not h.all() else np.nan
    fold_rows.append({"fold": k, "patients": len(set(oof_pat[sel])), "frames": len(sel), "nerve share": f"{h.mean():.2f}",
                      "balanced": f"{balanced_score(d, h):.3f}" if h.any() and not h.all() else "-",
                      "nerve Dice": f"{d[h].mean():.3f}" if h.any() else "-", "nerve found": f"{found[h].mean():.2f}" if h.any() else "-",
                      "empty kept": f"{(~found[~h]).mean():.2f}" if (~h).any() else "-", "head AUROC": f"{auc:.3f}"})
add_table("D1_folds", f"Table D1: out-of-fold results per fold (final configuration: {final_mode}, {final_kind})",
          "Large differences between folds mean the result depends on which patients are in the training data.\n"
          "Thresholds here were tuned on all folds, so these values are slightly optimistic.", pd.DataFrame(fold_rows))

pat_rows = []
for p in np.unique(test_pat):
    sel = np.where(test_pat == p)[0]
    h = test_has[sel]
    auc = roc_auc_score(h, fin_pres[sel]) if h.any() and not h.all() else np.nan
    pat_rows.append({"patient": int(p), "frames": len(sel), "nerve share": f"{h.mean():.2f}",
                     "nerve Dice": f"{fin_dice[sel][h].mean():.3f}" if h.any() else "-",
                     "nerve found": f"{fin_found[sel][h].mean():.2f}" if h.any() else "-",
                     "empty kept": f"{(~fin_found[sel][~h]).mean():.2f}" if (~h).any() else "-",
                     "head AUROC": f"{auc:.3f}" if np.isfinite(auc) else "-"})
add_table("D2_test_patients", "Table D2: test results per patient (final configuration, models averaged)", "", pd.DataFrame(pat_rows))


##### 9. Plots
saved_plots = []
def save_fig(fig, name):
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    saved_plots.append(path)

# P1: learning curves
hist_path = os.path.join(TABLE_DIR, "train_history.csv")
if os.path.exists(hist_path):
    hist = pd.read_csv(hist_path)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, col, title in [(axes[0], "train_loss", "training loss"), (axes[1], "val_loss", "loss on the held-out fold")]:
        for k, g in hist.groupby("fold"):
            ax.plot(g.epoch, g[col], label=f"fold {k}")
        ax.set_title(title); ax.set_xlabel("epoch")
    axes[0].set_ylabel("loss (pixel + Dice + 0.5 x head)"); axes[1].legend(frameon=False, fontsize=7)
    fig.suptitle("P1  Learning curves. A held-out curve that stops falling while training keeps falling = overfitting.", fontsize=9, y=1.04)
    save_fig(fig, "P1_learning_curves.png")

# P2: ROC and precision-recall of the detection head vs max pixel probability
test_ens_seg_flipmode, test_ens_pres_flipmode = fin_seg, fin_pres
curves = [("head, OOF", oof_pres[final_mode], oof_has, C_NERVE, "-"),
          (f"head, test ({N_FOLDS} avg)", test_ens_pres_flipmode, test_has, C_NERVE, "--"),
          ("max pixel prob., OOF", peak(oof_seg[final_mode]), oof_has, C_EMPTY, "-"),
          (f"max pixel prob., test ({N_FOLDS} avg)", peak(test_ens_seg_flipmode), test_has, C_EMPTY, "--")]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
for label, score, h, color, ls in curves:
    fpr, tpr, _ = roc_curve(h, score)
    axes[0].plot(fpr, tpr, color=color, ls=ls, label=f"{label}  AUROC {roc_auc_score(h, score):.2f}")
    prec, rec, _ = precision_recall_curve(h, score)
    axes[1].plot(rec, prec, color=color, ls=ls, label=f"{label}  AP {average_precision_score(h, score):.2f}")
axes[0].plot([0, 1], [0, 1], color=C_GREY, lw=0.8); axes[0].set_xlabel("false positive rate"); axes[0].set_ylabel("true positive rate")
axes[1].axhline(oof_has.mean(), color=C_GREY, lw=0.8); axes[1].set_xlabel("recall"); axes[1].set_ylabel("precision")
for ax in axes:
    ax.legend(frameon=False, fontsize=7, loc="lower right" if ax is axes[0] else "lower left")
fig.suptitle(f"P2  Detection of a nerve in a frame ({final_mode}). Dashed = test patients.", fontsize=9, y=1.04)
save_fig(fig, "P2_roc_pr.png")

# P3: histograms of the head probability
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=False)
for ax, title, prob, h in [(axes[0], "out-of-fold", oof_pres[final_mode], oof_has), (axes[1], f"test ({N_FOLDS} averaged)", fin_pres, test_has)]:
    bins = np.linspace(0, 1, 21)
    ax.hist(prob[~h], bins=bins, alpha=0.7, color=C_EMPTY, label="empty frames", density=True)
    ax.hist(prob[h], bins=bins, alpha=0.7, color=C_NERVE, label="nerve frames", density=True)
    ax.axvline(final_settings["pres_thr"], color="k", lw=1, ls="--")
    ax.set_title(title); ax.set_xlabel("detection head probability")
axes[0].set_ylabel("density"); axes[0].legend(frameon=False)
fig.suptitle("P3  Does the head separate nerve from empty frames? Dashed line = chosen threshold.", fontsize=9, y=1.04)
save_fig(fig, "P3_head_probability.png")

# P4: threshold sweep for the head
sweep_rows = []
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
for ax, title, prob, h in [(axes[0], "out-of-fold", oof_pres[final_mode], oof_has), (axes[1], f"test ({N_FOLDS} averaged)", fin_pres, test_has)]:
    thr = np.round(np.arange(0.05, 0.96, 0.05), 2)
    sens = np.array([(prob[h] >= t).mean() for t in thr])
    spec = np.array([(prob[~h] < t).mean() for t in thr])
    for t, a, b in zip(thr, sens, spec):
        sweep_rows.append({"data": title, "threshold": t, "sensitivity": round(float(a), 3), "specificity": round(float(b), 3),
                           "balanced accuracy": round(float((a + b) / 2), 3)})
    ax.plot(thr, sens, color=C_NERVE, label="sensitivity (nerve found)")
    ax.plot(thr, spec, color=C_EMPTY, label="specificity (empty kept empty)")
    ax.plot(thr, (sens + spec) / 2, color="k", lw=1, label="balanced accuracy")
    ax.axvline(final_settings["pres_thr"], color=C_GREY, ls="--", lw=0.8)
    ax.set_title(title); ax.set_xlabel("head threshold")
axes[0].legend(frameon=False, fontsize=7)
fig.suptitle("P4  Trade-off when the head decides 'nerve / no nerve'.", fontsize=9, y=1.04)
save_fig(fig, "P4_threshold_sweep.png")
pd.DataFrame(sweep_rows).to_csv(os.path.join(TABLE_DIR, "B3_threshold_sweep.csv"), index=False)

# P5: reliability diagram
fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), sharey=True)
for ax, title, prob, h in [(axes[0], "out-of-fold", oof_pres[final_mode], oof_has), (axes[1], f"test ({N_FOLDS} averaged)", fin_pres, test_has)]:
    edges = np.linspace(0, 1, 11)
    which = np.clip(np.digitize(prob, edges) - 1, 0, 9)
    xs = [prob[which == b].mean() for b in range(10) if (which == b).any()]
    ys = [h[which == b].mean() for b in range(10) if (which == b).any()]
    ns = [(which == b).sum() for b in range(10) if (which == b).any()]
    ax.plot([0, 1], [0, 1], color=C_GREY, lw=0.8)
    ax.scatter(xs, ys, s=[20 + 300 * n / max(ns) for n in ns], color=C_NERVE, alpha=0.8)
    ax.plot(xs, ys, color=C_NERVE, lw=0.8)
    ax.set_title(f"{title}  (ECE {ece(prob, h.astype(float)):.2f})"); ax.set_xlabel("predicted probability")
axes[0].set_ylabel("share of frames with a nerve")
fig.suptitle("P5  Calibration of the head. On the diagonal = probabilities can be read as probabilities. Dot size = frames.", fontsize=9, y=1.04)
save_fig(fig, "P5_calibration.png")

# P6: configurations compared (test rows) with intervals
test_rows = config_df[config_df.dataset == "test (10 pt)"].reset_index(drop=True)
labels = [f"{r.models} | {r.flip} | {r.settings}" for r in test_rows.itertuples()]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, col, title in [(axes[0], "balanced", "balanced score"), (axes[1], "nerve_dice", "Dice on nerve frames")]:
    y = np.arange(len(test_rows))[::-1]
    err = [test_rows[col] - test_rows[col + "_lo"], test_rows[col + "_hi"] - test_rows[col]]
    ax.errorbar(test_rows[col], y, xerr=err, fmt="o", color=C_NERVE, ecolor=C_GREY, capsize=2)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7); ax.set_title(title)
    if col == "balanced":
        ax.axvline(0.5, color=C_RED, ls="--", lw=0.8); ax.text(0.5, y.max() + 0.6, " always empty", color=C_RED, fontsize=7)
fig.suptitle("P6  Test patients: every way of using the models, with 95% intervals over patients.", fontsize=9, y=1.04)
save_fig(fig, "P6_configurations.png")

# P7: Dice against nerve size (test nerve frames, final configuration)
area = test_true[test_has].sum(axis=(1, 2))
dice_nerve = fin_dice[test_has]
fig, ax = plt.subplots(figsize=(5.5, 3.8))
ax.scatter(area, dice_nerve, s=8, alpha=0.4, color=C_NERVE)
edges = np.quantile(area, np.linspace(0, 1, 6))
mid, med = [], []
for a, b in zip(edges[:-1], edges[1:]):
    sel = (area >= a) & (area <= b)
    if sel.any():
        mid.append(np.median(area[sel])); med.append(np.median(dice_nerve[sel]))
ax.plot(mid, med, "o-", color=C_RED, label="median per size quintile")
ax.set_xlabel(f"true nerve area (pixels on the {H}x{W} grid)"); ax.set_ylabel("Dice"); ax.legend(frameon=False)
ax.set_title("P7  Is a small nerve harder to find?", fontsize=9)
save_fig(fig, "P7_dice_vs_size.png")

# P8: per fold and per patient
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
fold_vals = []
for k in range(N_FOLDS):
    sel = np.where(fold_of_dev == k)[0]
    h = oof_has[sel]
    fold_vals.append(balanced_score(frame_dice(oof_final_masks[sel], oof_true[sel]), h) if h.any() and not h.all() else np.nan)
axes[0].bar(range(N_FOLDS), fold_vals, color=C_NERVE)
axes[0].axhline(0.5, color=C_RED, ls="--", lw=0.8); axes[0].set_xlabel("fold"); axes[0].set_ylabel("balanced score")
axes[0].set_title("out-of-fold, per fold")
pats = np.unique(test_pat)
nd = [fin_dice[(test_pat == p) & test_has].mean() if ((test_pat == p) & test_has).any() else np.nan for p in pats]
fd = [fin_found[(test_pat == p) & test_has].mean() if ((test_pat == p) & test_has).any() else np.nan for p in pats]
x = np.arange(len(pats))
axes[1].bar(x - 0.2, nd, 0.4, color=C_NERVE, label="Dice on nerve frames")
axes[1].bar(x + 0.2, fd, 0.4, color=C_EMPTY, label="share of nerve frames found")
axes[1].set_xticks(x); axes[1].set_xticklabels([str(p) for p in pats]); axes[1].set_xlabel("test patient"); axes[1].legend(frameon=False, fontsize=7)
axes[1].set_title("test, per patient")
fig.suptitle("P8  Spread between folds and between patients.", fontsize=9, y=1.04)
save_fig(fig, "P8_folds_and_patients.png")

# P9: why flip averaging and averaging 5 models can hurt
peak_sets = {
    "1 model\nno flip": np.concatenate([peak(s[test_has]) for s, _ in test_preds["no flip"]]),
    "1 model\nflip": np.concatenate([peak(s[test_has]) for s, _ in test_preds["flip"]]),
    f"{N_FOLDS} avg\nno flip": peak(average_models(test_preds["no flip"])[0][test_has]),
    f"{N_FOLDS} avg\nflip": peak(average_models(test_preds["flip"])[0][test_has]),
}
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
axes[0].boxplot(list(peak_sets.values()), showfliers=False)
axes[0].set_xticks(range(1, len(peak_sets) + 1))
axes[0].set_xticklabels(list(peak_sets.keys()), fontsize=7)
axes[0].axhline(0.5, color=C_RED, ls="--", lw=0.8)
axes[0].set_ylabel("highest pixel probability"); axes[0].set_title("how confident is the model on nerve frames?")
valid = pair_dice[np.isfinite(pair_dice)]
axes[1].hist(valid, bins=np.linspace(0, 1, 21), color=C_NERVE)
axes[1].set_xlabel("Dice between mask of the image and mask of its mirror image")
axes[1].set_ylabel("test nerve frames"); axes[1].set_title(f"flip consistency (median {np.median(valid):.2f})")
fig.suptitle("P9  Averaging with a mirrored prediction only helps if the model gives the same answer for both. Red line = 0.5.", fontsize=9, y=1.04)
save_fig(fig, "P9_flip_and_ensemble.png")

# P10: examples - worst, median, best nerve frames and false alarms (final configuration, test patients)
nerve_pos = np.where(test_has)[0]
order = nerve_pos[np.argsort(fin_dice[nerve_pos])]
mid_start = max(0, len(order) // 2 - 2)
fp_pos = np.where(~test_has & fin_found)[0]
groups_to_show = [("worst", order[:4]), ("median", order[mid_start:mid_start + 4]), ("best", order[-4:]),
                  ("false alarm", fp_pos[rng.permutation(len(fp_pos))[:4]])]
groups_to_show = [(name, ids) for name, ids in groups_to_show if len(ids) > 0]     # no empty rows
fig, axes = plt.subplots(len(groups_to_show), 4, figsize=(9, 2.3 * len(groups_to_show)), squeeze=False)
for r, (name, ids) in enumerate(groups_to_show):
    for c in range(4):
        ax = axes[r, c]
        ax.axis("off")
        if c < len(ids):
            i = ids[c]
            ax.imshow(X[test_idx[i]], cmap="gray", vmin=0, vmax=1)
            if test_true[i].any():
                ax.contour(test_true[i].astype(float), levels=[0.5], colors="#2ecc71", linewidths=1.2)
            if fin_masks[i].any():
                ax.contour(fin_masks[i].astype(float), levels=[0.5], colors="#e74c3c", linewidths=1.2)
            ax.set_title(f"{name}: Dice {fin_dice[i]:.2f}, head {fin_pres[i]:.2f}", fontsize=7)
fig.suptitle("P10  Test frames. Green = true nerve, red = prediction. Rows: worst, median, best, false alarms.", fontsize=9, y=1.04)
save_fig(fig, "P10_examples.png")


##### 10. Report, config for C++, MLflow

with open(os.path.join(TABLE_DIR, "report.md"), "w") as f:
    f.write("# Diagnostics report (step 9)\n\n")
    f.write(f"Final configuration, chosen on out-of-fold predictions only: flip averaging = {final_mode}, "
            f"thresholds = {final_kind} {final_settings}\n\n")
    for title, note, df in report_parts:
        f.write(f"## {title}\n\n")
        if note:
            f.write(note.replace("\n", "  \n") + "\n\n")
        f.write(md_table(df) + "\n\n")

post = []
if final_config["flip_average"]:
    post += ["apply sigmoid to both outputs", "do the same for the mirrored input (flip left-right), flip the segmentation back, average the two"]
else:
    post += ["apply sigmoid to both outputs (no mirrored input)"]
post += [f"average over all {N_FOLDS} models",
         f"if the presence probability is below {final_settings['pres_thr']}: empty mask",
         f"set pixels with probability above {final_settings['pix_thr']} to nerve"]
if final_largest:
    post += ["keep only the largest connected blob"]
post += [f"if the blob has fewer than {final_settings['min_area']} pixels: empty mask"]
cpp_config = {
    "models": [f"nerve_unet_fold{k}_torchscript.pt" for k in range(N_FOLDS)],
    "input_shape": [1, 1, H, W],
    "preprocessing": ["load the image as grayscale", f"resize to {W}x{H} (width x height) with bilinear interpolation (PIL Image.BILINEAR)",
                      "divide by 255, so values are float32 between 0 and 1"],
    "model_outputs": [f"segmentation logits, shape (1, 1, {H}, {W})", "presence logit, shape (1)"],
    "postprocessing": post,
    "thresholds": final_settings,
    "chosen_on": "out-of-fold predictions of the non-test patients only",
    "note": "Run models/nerve_reference_input.bin through libtorch and compare with the reference outputs before trusting the export."}
with open(os.path.join(OUT_DIR, "models", "nerve_unet_config.json"), "w") as f:
    json.dump(cpp_config, f, indent=2)

final_row = config_df[(config_df.dataset == "test (10 pt)") & (config_df.models == f"{N_FOLDS} averaged")
                      & (config_df.flip == final_mode) & (config_df.settings == ("default" if final_kind == "default" else "tuned"))].iloc[0]
for name in ["balanced", "nerve_dice", "found", "empty_kept"]:
    mlflow.log_metric(f"test_{name}", float(final_row[name]))
    mlflow.log_metric(f"test_{name}_ci_low", float(final_row[name + "_lo"]))
    mlflow.log_metric(f"test_{name}_ci_high", float(final_row[name + "_hi"]))
mlflow.log_params({"final_flip": final_mode, "final_thresholds": final_kind, **{f"final_{k}": v for k, v in final_settings.items()}})
mlflow.log_artifacts(TABLE_DIR, artifact_path="tables")
mlflow.log_artifacts(PLOT_DIR, artifact_path="plots")
mlflow.log_artifact(os.path.join(OUT_DIR, "final_config.json"))
try:
    mlflow.log_artifact(__file__)
except NameError:
    pass
mlflow.end_run()
print(f"\nsaved {len(saved_plots)} plots to {PLOT_DIR} and tables to {TABLE_DIR} (report.md has all tables)")
print("next: run step10_explain.py")##### Step 9: diagnostics (tables and plots)
# Reads the 5 models and the split saved by step8_train.py. Trains nothing.
# It answers four questions:
#   A. Which way of using the models is best: flip averaging on/off, default or tuned thresholds,
#      one model or 5 models? (step 7 found that flip + 5 models finds almost no nerve)
#   B. How good is the detection ("is there a nerve?") on its own? ROC, PR, calibration.
#   C. How good is the segmentation when there is a nerve? Dice, IoU, precision, recall, size effect.
#   D. How much do the numbers move between folds and between patients?
# Every number comes with a 95% interval. The intervals redraw whole PATIENTS (not frames),
# because frames of one patient are not independent.
#
# Rule used here: nothing is chosen on the test patients. The configuration is chosen on the
# out-of-fold (OOF) predictions of the 37 non-test patients. The test patients are only reported.
# "tuned (cross-fitted)" means: thresholds are tuned on 4 folds and applied to the 5th, so the
# score is not inflated by tuning and scoring on the same frames.
# Simple options win ties: tuned thresholds or flip averaging are only chosen if they beat the
# simplest option (no flip, default thresholds) by at least 0.005 balanced score out-of-fold.
#
# Output (in OUT_DIR): tables/*.csv, tables/report.md, plots/*.png, final_config.json,
#                      models/nerve_unet_config.json (for the C++ side)
# Time: a few minutes (the threshold search is the slow part).
# Needs:  pip install mlflow torch scikit-learn scipy pandas matplotlib

import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import ndimage
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

OUT_DIR = os.environ.get("OUT_DIR", "outputs")
H, W = 96, 128
SEED = 0
N_BOOT = int(os.environ.get("N_BOOT", 2000))          # patient redraws for most intervals
N_BOOT_AUC = int(os.environ.get("N_BOOT_AUC", 500))   # for AUROC / AP (slower per redraw)
PRES_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]    # values to try for the detection threshold
PIX_GRID = [0.3, 0.4, 0.5, 0.6]                          # values to try for the pixel threshold
AREA_GRID = [0, 25, 50, 100, 200, 300, 400]              # smallest allowed blob, in pixels on the 96x128 grid
RUN_NAME = "step9_diagnostics"

TABLE_DIR = os.path.join(OUT_DIR, "tables")
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
rng = np.random.default_rng(SEED)

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 130, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
C_NERVE, C_EMPTY, C_GREY, C_RED = "#1b6ca8", "#d9822b", "#8a8a8a", "#c0392b"

mlflow.set_tracking_uri(f"sqlite:///{os.path.join(OUT_DIR, 'mlflow.db')}")
mlflow.set_experiment("nerve_segmentation")
mlflow.start_run(run_name=RUN_NAME)
mlflow.log_params({"step": 9, "n_boot": N_BOOT, "n_boot_auc": N_BOOT_AUC})


##### 1. Load the cache, the split and the 5 models

cache = np.load(os.path.join(OUT_DIR, "data_cache.npz"))
X = cache["X"].astype(np.float32) / 255.0
Y = cache["Y"].astype(np.float32)
patients = cache["patients"]
has_nerve = Y.sum(axis=(1, 2)) > 0
split = np.load(os.path.join(OUT_DIR, "split.npz"))
dev_idx, test_idx, fold_of_dev = split["dev_idx"], split["test_idx"], split["fold_of_dev"]
N_FOLDS = int(fold_of_dev.max()) + 1

oof_true, oof_has, oof_pat = Y[dev_idx] > 0.5, has_nerve[dev_idx], patients[dev_idx]
test_true, test_has, test_pat = Y[test_idx] > 0.5, has_nerve[test_idx], patients[test_idx]
print(f"dev: {len(dev_idx)} frames, {len(set(oof_pat))} patients   test: {len(test_idx)} frames, {len(set(test_pat))} patients")

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
        self.presence = nn.Linear(64 * 2, 1)

    def forward(self, x, return_features=False):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        m = self.middle(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(m), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        pooled = torch.cat([m.mean(dim=(2, 3)), m.amax(dim=(2, 3))], dim=1)
        if return_features:
            return self.out(d1), self.presence(pooled)[:, 0], m
        return self.out(d1), self.presence(pooled)[:, 0]

models = []
for k in range(N_FOLDS):
    model = UNet()
    model.load_state_dict(torch.load(os.path.join(OUT_DIR, "models", f"unet_fold{k}.pt"), map_location="cpu"))
    models.append(model.to(device).eval())


##### 2. Predict: out-of-fold for the dev patients, and all 5 models for the test patients
# Both modes are kept: "no flip" and "flip" (the input and its mirror image are predicted and averaged).

def predict(model, ids, flip_average):
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

def predict_pair(model, ids):
    """Mask of the normal input and mask of the mirrored input (flipped back), both at 0.5."""
    a, b = [], []
    with torch.no_grad():
        for start in range(0, len(ids), 64):
            xb = torch.from_numpy(X[ids[start:start + 64]]).unsqueeze(1).to(device)
            a.append((torch.sigmoid(model(xb)[0])[:, 0] > 0.5).cpu().numpy())
            b.append((torch.sigmoid(model(xb.flip(-1))[0]).flip(-1)[:, 0] > 0.5).cpu().numpy())
    return np.concatenate(a), np.concatenate(b)

MODES = ["no flip", "flip"]
oof_seg = {m: np.zeros((len(dev_idx), H, W), dtype=np.float16) for m in MODES}
oof_pres = {m: np.zeros(len(dev_idx), dtype=np.float32) for m in MODES}
test_preds = {m: [] for m in MODES}                 # one (segmentation, presence) pair per model
pair_dice = []                                      # flip consistency per model
t0 = time.time()
for k, model in enumerate(models):
    val = np.where(fold_of_dev == k)[0]
    for mode in MODES:
        s, p = predict(model, dev_idx[val], flip_average=(mode == "flip"))
        oof_seg[mode][val], oof_pres[mode][val] = s, p
        test_preds[mode].append(predict(model, test_idx, flip_average=(mode == "flip")))
    a, b = predict_pair(model, test_idx[test_has])
    inter = (a & b).sum(axis=(1, 2))
    total = a.sum(axis=(1, 2)) + b.sum(axis=(1, 2))
    pair_dice.append(np.where(total == 0, np.nan, 2 * inter / np.maximum(total, 1)))
    print(f"  model {k + 1}/{N_FOLDS} predicted")
pair_dice = np.nanmean(np.stack(pair_dice), axis=0)       # per test nerve frame, mean over models
print(f"predictions took {time.time() - t0:.0f} s")


##### 3. Masks, scores and the patient bootstrap

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

def frame_dice(pred, true):
    overlap = (pred & true).sum(axis=(1, 2))
    total = pred.sum(axis=(1, 2)) + true.sum(axis=(1, 2))
    return np.where(total == 0, 1.0, 2 * overlap / np.maximum(total, 1))

def frame_overlap(pred, true):
    """Dice, IoU, precision, recall per frame (meant for frames that contain a nerve)."""
    inter = (pred & true).sum(axis=(1, 2)).astype(np.float64)
    p, t = pred.sum(axis=(1, 2)), true.sum(axis=(1, 2))
    return {"dice": np.where(p + t == 0, 1.0, 2 * inter / np.maximum(p + t, 1)),
            "iou": np.where(p + t == 0, 1.0, inter / np.maximum(p + t - inter, 1)),
            "precision": np.where(p == 0, 0.0, inter / np.maximum(p, 1)),
            "recall": np.where(t == 0, 0.0, inter / np.maximum(t, 1))}

def balanced_score(dice, frame_has_nerve):
    return 0.5 * dice[frame_has_nerve].mean() + 0.5 * dice[~frame_has_nerve].mean()

def average_models(predictions):
    seg = np.mean([p[0] for p in predictions], axis=0, dtype=np.float32)
    pres = np.mean([p[1] for p in predictions], axis=0)
    return seg, pres

def patient_bootstrap(stat_fns, groups, n_boot):
    """Redraw whole patients with replacement. stat_fns: {name: function(frame_index_array) -> number}.
    Returns {name: (low, high)}, the middle 95% of the redrawn values."""
    draw_rng = np.random.default_rng(SEED)
    ids = np.unique(groups)
    members = [np.where(groups == g)[0] for g in ids]
    values = {name: [] for name in stat_fns}
    for _ in range(n_boot):
        drawn = draw_rng.integers(0, len(ids), len(ids))
        idx = np.concatenate([members[j] for j in drawn])
        for name, fn in stat_fns.items():
            values[name].append(fn(idx))
    out = {}
    for name, v in values.items():
        v = np.array(v, dtype=float)
        out[name] = (np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)) if np.isfinite(v).any() else (np.nan, np.nan)
    return out

def evaluate(mask_list, true, has, groups):
    """Detection+segmentation score of one configuration.
    mask_list holds one mask array per model (one entry for out-of-fold rows). Results are averaged over the list."""
    d_list = [frame_dice(m, true) for m in mask_list]
    f_list = [m.any(axis=(1, 2)) for m in mask_list]

    def nerve_dice(idx):
        h = has[idx]
        return np.nan if not h.any() else float(np.mean([d[idx][h].mean() for d in d_list]))
    def found(idx):
        h = has[idx]
        return np.nan if not h.any() else float(np.mean([f[idx][h].mean() for f in f_list]))
    def empty_kept(idx):
        h = has[idx]
        return np.nan if h.all() else float(np.mean([(~f[idx][~h]).mean() for f in f_list]))
    def balanced(idx):
        return 0.5 * nerve_dice(idx) + 0.5 * empty_kept(idx)

    fns = {"balanced": balanced, "nerve_dice": nerve_dice, "found": found, "empty_kept": empty_kept}
    everything = np.arange(len(has))
    ci = patient_bootstrap(fns, groups, N_BOOT)
    row = {}
    for name, fn in fns.items():
        row[name], row[name + "_lo"], row[name + "_hi"] = fn(everything), ci[name][0], ci[name][1]
    return row

def fmt(value, lo, hi):
    return f"{value:.3f} [{lo:.3f}, {hi:.3f}]"

def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)

report_parts = []          # (title, note, dataframe) for report.md
def add_table(name, title, note, df_numeric, df_display=None):
    df_numeric.to_csv(os.path.join(TABLE_DIR, name + ".csv"), index=False)
    shown = df_display if df_display is not None else df_numeric
    report_parts.append((title, note, shown))
    print(f"\n=== {title} ===")
    if note:
        print(note)
    print(shown.to_string(index=False))


##### 4. Thresholds: default, tuned on all OOF frames, and tuned cross-fitted
# The search is the one of steps 5 to 7: one threshold at a time, best balanced score wins.

DEFAULT = {"pres_thr": 0.5, "pix_thr": 0.5, "min_area": 0}

def tune(seg, pres, true, has):
    best = dict(DEFAULT)
    for name, grid in [("pres_thr", PRES_GRID), ("pix_thr", PIX_GRID), ("min_area", AREA_GRID)]:
        scores = []
        for value in grid:
            trial = {**best, name: value}
            masks = make_masks(seg, pres, trial["pres_thr"], trial["pix_thr"], trial["min_area"], True)
            scores.append(balanced_score(frame_dice(masks, true), has))
        best[name] = grid[int(np.argmax(scores))]
    return best

tuned, cross_fit_masks = {}, {}
t0 = time.time()
for mode in MODES:
    tuned[mode] = tune(oof_seg[mode], oof_pres[mode], oof_true, oof_has)
    cf = np.zeros((len(dev_idx), H, W), dtype=bool)
    for k in range(N_FOLDS):
        fit, apply_to = fold_of_dev != k, fold_of_dev == k
        t = tune(oof_seg[mode][fit], oof_pres[mode][fit], oof_true[fit], oof_has[fit])
        cf[apply_to] = make_masks(oof_seg[mode][apply_to], oof_pres[mode][apply_to], t["pres_thr"], t["pix_thr"], t["min_area"], True)
    cross_fit_masks[mode] = cf
    print(f"tuned thresholds ({mode}): {tuned[mode]}")
print(f"threshold search took {time.time() - t0:.0f} s")


##### 5. Table A: the configuration grid

rows = []
def add_row(dataset, models_used, mode, settings_name, res):
    rows.append({"dataset": dataset, "models": models_used, "flip": mode, "settings": settings_name, **res})

empty_oof = [np.zeros_like(oof_true)]
empty_test = [np.zeros_like(test_true)]
add_row("OOF (37 pt)", "-", "-", "always empty", evaluate(empty_oof, oof_true, oof_has, oof_pat))
for mode in MODES:
    seg, pres = oof_seg[mode], oof_pres[mode]
    add_row("OOF (37 pt)", "1 (out-of-fold)", mode, "default",
            evaluate([make_masks(seg, pres, 0.5, 0.5, 0, False)], oof_true, oof_has, oof_pat))
    add_row("OOF (37 pt)", "1 (out-of-fold)", mode, "tuned, same frames (optimistic)",
            evaluate([make_masks(seg, pres, tuned[mode]["pres_thr"], tuned[mode]["pix_thr"], tuned[mode]["min_area"], True)],
                     oof_true, oof_has, oof_pat))
    add_row("OOF (37 pt)", "1 (out-of-fold)", mode, "tuned, cross-fitted",
            evaluate([cross_fit_masks[mode]], oof_true, oof_has, oof_pat))

add_row("test (10 pt)", "-", "-", "always empty", evaluate(empty_test, test_true, test_has, test_pat))
for mode in MODES:
    for settings_name, t, largest in [("default", DEFAULT, False), ("tuned", tuned[mode], True)]:
        single = [make_masks(s, p, t["pres_thr"], t["pix_thr"], t["min_area"], largest) for s, p in test_preds[mode]]
        add_row("test (10 pt)", f"1 (mean of {N_FOLDS})", mode, settings_name, evaluate(single, test_true, test_has, test_pat))
        ens_seg, ens_pres = average_models(test_preds[mode])
        add_row("test (10 pt)", f"{N_FOLDS} averaged", mode, settings_name,
                evaluate([make_masks(ens_seg, ens_pres, t["pres_thr"], t["pix_thr"], t["min_area"], largest)],
                         test_true, test_has, test_pat))

config_df = pd.DataFrame(rows)
display = config_df[["dataset", "models", "flip", "settings"]].copy()
for name, title in [("balanced", "balanced score"), ("nerve_dice", "nerve-frame Dice"),
                    ("found", "nerve found"), ("empty_kept", "empty kept empty")]:
    display[title] = [fmt(r[name], r[name + "_lo"], r[name + "_hi"]) for _, r in config_df.iterrows()]
add_table("A_configurations", "Table A: configurations (value [95% interval over patients])",
          "balanced score = half the Dice on nerve frames + half the Dice on empty frames. 'Always empty' scores 0.5.\n"
          "OOF rows: every frame is predicted by a model that never saw its patient.\n"
          "Test rows: the test patients were not used for training or for choosing anything.", config_df, display)

# Choose the final configuration on OOF only, using the honest value (cross-fitted for tuned).
candidates = {}
for mode in MODES:
    d = config_df[(config_df.dataset == "OOF (37 pt)") & (config_df.flip == mode)]
    candidates[(mode, "default")] = float(d[d.settings == "default"].balanced.iloc[0])
    candidates[(mode, "tuned")] = float(d[d.settings == "tuned, cross-fitted"].balanced.iloc[0])
# Simple options come first. A more complex option (tuned thresholds, flip averaging) must win by
# at least MARGIN, otherwise a difference of 0.001 would decide the configuration.
MARGIN = 0.005
final_mode, final_kind = ("no flip", "default")
for option in [("no flip", "tuned"), ("flip", "default"), ("flip", "tuned")]:
    if candidates[option] >= candidates[(final_mode, final_kind)] + MARGIN:
        final_mode, final_kind = option
final_settings = dict(DEFAULT) if final_kind == "default" else dict(tuned[final_mode])
final_largest = final_kind == "tuned"
print(f"\nchosen on OOF only: flip = {final_mode}, thresholds = {final_kind} {final_settings} "
      f"(OOF balanced {candidates[(final_mode, final_kind)]:.3f})")
final_config = {"flip_average": final_mode == "flip", "thresholds": final_kind, "settings": final_settings,
                "largest_blob_only": final_largest, "candidates_oof_balanced": {f"{m} / {s}": v for (m, s), v in candidates.items()},
                "ensemble": f"average of the {N_FOLDS} fold models (design choice, see Table A for its test result)"}
with open(os.path.join(OUT_DIR, "final_config.json"), "w") as f:
    json.dump(final_config, f, indent=2)

# the final pipeline on the test patients
fin_seg, fin_pres = average_models(test_preds[final_mode])
fin_masks = make_masks(fin_seg, fin_pres, final_settings["pres_thr"], final_settings["pix_thr"],
                       final_settings["min_area"], final_largest)
fin_dice = frame_dice(fin_masks, test_true)
fin_found = fin_masks.any(axis=(1, 2))
oof_final_masks = make_masks(oof_seg[final_mode], oof_pres[final_mode], final_settings["pres_thr"],
                             final_settings["pix_thr"], final_settings["min_area"], final_largest)


##### 6. Table B: detection ("is there a nerve in this frame?")

def ece(prob, label, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(prob, edges) - 1, 0, bins - 1)
    return float(sum((which == b).mean() * abs(prob[which == b].mean() - label[which == b].mean())
                     for b in range(bins) if (which == b).any()))

def det_row(name, score_list, has, groups, thr_list):
    """score_list: one score array per model (or one). Values are averaged over the list."""
    def auc(idx):
        h = has[idx]
        return np.nan if h.all() or not h.any() else float(np.mean([roc_auc_score(h, s[idx]) for s in score_list]))
    def ap(idx):
        h = has[idx]
        return np.nan if not h.any() else float(np.mean([average_precision_score(h, s[idx]) for s in score_list]))
    everything = np.arange(len(has))
    ci = patient_bootstrap({"auroc": auc, "ap": ap}, groups, N_BOOT_AUC)
    out = {"scores": name, "AUROC": fmt(auc(everything), *ci["auroc"]), "AP": fmt(ap(everything), *ci["ap"]),
           "prevalence": f"{has.mean():.2f}"}
    for thr in thr_list:
        sens = np.mean([(s[has] >= thr).mean() for s in score_list])
        spec = np.mean([(s[~has] < thr).mean() for s in score_list])
        out[f"sens/spec @{thr:.2f}"] = f"{sens:.2f} / {spec:.2f}"
    return out

def peak(seg):                                          # highest pixel probability per frame
    return seg.reshape(len(seg), -1).max(axis=1).astype(np.float32)

det_rows = []
thr_pair = [0.5, final_settings["pres_thr"]] if final_settings["pres_thr"] != 0.5 else [0.5]
for mode in MODES:
    det_rows.append(det_row(f"OOF, detection head, {mode}", [oof_pres[mode]], oof_has, oof_pat, thr_pair))
    det_rows.append(det_row(f"OOF, max pixel probability, {mode}", [peak(oof_seg[mode])], oof_has, oof_pat, [0.5]))
for mode in MODES:
    ens_seg, ens_pres = average_models(test_preds[mode])
    det_rows.append(det_row(f"test, head, {N_FOLDS} averaged, {mode}", [ens_pres], test_has, test_pat, thr_pair))
    det_rows.append(det_row(f"test, head, 1 model (mean), {mode}", [p for _, p in test_preds[mode]], test_has, test_pat, thr_pair))
    det_rows.append(det_row(f"test, max pixel prob., {N_FOLDS} averaged, {mode}", [peak(ens_seg)], test_has, test_pat, [0.5]))
det_df = pd.DataFrame(det_rows).fillna("-")
add_table("B_detection", "Table B: frame-level detection of a nerve",
          "AUROC and AP do not depend on a threshold. 'max pixel probability' uses the segmentation output instead of the head.\n"
          "If the head is not clearly better than the max pixel probability, the extra head adds little.", det_df)

# calibration of the head (final mode)
ens_pres_final = fin_pres
cal_rows = []
for name, prob, label in [("OOF", oof_pres[final_mode], oof_has), (f"test ({N_FOLDS} averaged)", ens_pres_final, test_has)]:
    cal_rows.append({"data": name, "ECE (10 bins)": f"{ece(prob, label.astype(float)):.3f}",
                     "mean prob. on nerve frames": f"{prob[label].mean():.2f}",
                     "mean prob. on empty frames": f"{prob[~label].mean():.2f}"})
add_table("B2_calibration", f"Table B2: calibration of the detection head ({final_mode})",
          "ECE = average gap between predicted probability and the real share of nerve frames (0 is perfect).",
          pd.DataFrame(cal_rows))


##### 7. Table C: segmentation quality on frames that contain a nerve
# Detection gating and clean-up are switched off here (pixel threshold 0.5), so this is the segmentation alone.

def seg_row(name, mask_list, true, has, groups):
    nerve = np.where(has)[0]
    per_model = [frame_overlap(m[nerve], true[nerve]) for m in mask_list]
    grp = groups[nerve]
    fns = {key: (lambda idx, key=key: float(np.mean([pm[key][idx].mean() for pm in per_model]))) for key in per_model[0]}
    ci = patient_bootstrap(fns, grp, N_BOOT)
    out = {"predictions": name}
    for key, fn in fns.items():
        out[key] = fmt(fn(np.arange(len(nerve))), *ci[key])
    return out

seg_rows = []
for mode in MODES:
    seg_rows.append(seg_row(f"OOF, 1 model, {mode}", [oof_seg[mode] > 0.5], oof_true, oof_has, oof_pat))
for mode in MODES:
    seg_rows.append(seg_row(f"test, 1 model (mean), {mode}", [s > 0.5 for s, _ in test_preds[mode]], test_true, test_has, test_pat))
    seg_rows.append(seg_row(f"test, {N_FOLDS} averaged, {mode}", [average_models(test_preds[mode])[0] > 0.5], test_true, test_has, test_pat))
add_table("C_segmentation", "Table C: segmentation quality on nerve frames only (no gating, pixel threshold 0.5)",
          "Dice/IoU measure the overlap. Precision = share of predicted pixels that are nerve. Recall = share of nerve pixels found.",
          pd.DataFrame(seg_rows))


##### 8. Table D: folds and patients

fold_rows = []
for k in range(N_FOLDS):
    sel = np.where(fold_of_dev == k)[0]
    d = frame_dice(oof_final_masks[sel], oof_true[sel])
    h = oof_has[sel]
    found = oof_final_masks[sel].any(axis=(1, 2))
    auc = roc_auc_score(h, oof_pres[final_mode][sel]) if h.any() and not h.all() else np.nan
    fold_rows.append({"fold": k, "patients": len(set(oof_pat[sel])), "frames": len(sel), "nerve share": f"{h.mean():.2f}",
                      "balanced": f"{balanced_score(d, h):.3f}" if h.any() and not h.all() else "-",
                      "nerve Dice": f"{d[h].mean():.3f}" if h.any() else "-", "nerve found": f"{found[h].mean():.2f}" if h.any() else "-",
                      "empty kept": f"{(~found[~h]).mean():.2f}" if (~h).any() else "-", "head AUROC": f"{auc:.3f}"})
add_table("D1_folds", f"Table D1: out-of-fold results per fold (final configuration: {final_mode}, {final_kind})",
          "Large differences between folds mean the result depends on which patients are in the training data.\n"
          "Thresholds here were tuned on all folds, so these values are slightly optimistic.", pd.DataFrame(fold_rows))

pat_rows = []
for p in np.unique(test_pat):
    sel = np.where(test_pat == p)[0]
    h = test_has[sel]
    auc = roc_auc_score(h, fin_pres[sel]) if h.any() and not h.all() else np.nan
    pat_rows.append({"patient": int(p), "frames": len(sel), "nerve share": f"{h.mean():.2f}",
                     "nerve Dice": f"{fin_dice[sel][h].mean():.3f}" if h.any() else "-",
                     "nerve found": f"{fin_found[sel][h].mean():.2f}" if h.any() else "-",
                     "empty kept": f"{(~fin_found[sel][~h]).mean():.2f}" if (~h).any() else "-",
                     "head AUROC": f"{auc:.3f}" if np.isfinite(auc) else "-"})
add_table("D2_test_patients", "Table D2: test results per patient (final configuration, models averaged)", "", pd.DataFrame(pat_rows))


##### 9. Plots
saved_plots = []
def save_fig(fig, name):
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    saved_plots.append(path)

# P1: learning curves
hist_path = os.path.join(TABLE_DIR, "train_history.csv")
if os.path.exists(hist_path):
    hist = pd.read_csv(hist_path)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, col, title in [(axes[0], "train_loss", "training loss"), (axes[1], "val_loss", "loss on the held-out fold")]:
        for k, g in hist.groupby("fold"):
            ax.plot(g.epoch, g[col], label=f"fold {k}")
        ax.set_title(title); ax.set_xlabel("epoch")
    axes[0].set_ylabel("loss (pixel + Dice + 0.5 x head)"); axes[1].legend(frameon=False, fontsize=7)
    fig.suptitle("P1  Learning curves. A held-out curve that stops falling while training keeps falling = overfitting.", fontsize=9, y=1.04)
    save_fig(fig, "P1_learning_curves.png")

# P2: ROC and precision-recall of the detection head vs max pixel probability
test_ens_seg_flipmode, test_ens_pres_flipmode = fin_seg, fin_pres
curves = [("head, OOF", oof_pres[final_mode], oof_has, C_NERVE, "-"),
          (f"head, test ({N_FOLDS} avg)", test_ens_pres_flipmode, test_has, C_NERVE, "--"),
          ("max pixel prob., OOF", peak(oof_seg[final_mode]), oof_has, C_EMPTY, "-"),
          (f"max pixel prob., test ({N_FOLDS} avg)", peak(test_ens_seg_flipmode), test_has, C_EMPTY, "--")]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
for label, score, h, color, ls in curves:
    fpr, tpr, _ = roc_curve(h, score)
    axes[0].plot(fpr, tpr, color=color, ls=ls, label=f"{label}  AUROC {roc_auc_score(h, score):.2f}")
    prec, rec, _ = precision_recall_curve(h, score)
    axes[1].plot(rec, prec, color=color, ls=ls, label=f"{label}  AP {average_precision_score(h, score):.2f}")
axes[0].plot([0, 1], [0, 1], color=C_GREY, lw=0.8); axes[0].set_xlabel("false positive rate"); axes[0].set_ylabel("true positive rate")
axes[1].axhline(oof_has.mean(), color=C_GREY, lw=0.8); axes[1].set_xlabel("recall"); axes[1].set_ylabel("precision")
for ax in axes:
    ax.legend(frameon=False, fontsize=7, loc="lower right" if ax is axes[0] else "lower left")
fig.suptitle(f"P2  Detection of a nerve in a frame ({final_mode}). Dashed = test patients.", fontsize=9, y=1.04)
save_fig(fig, "P2_roc_pr.png")

# P3: histograms of the head probability
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=False)
for ax, title, prob, h in [(axes[0], "out-of-fold", oof_pres[final_mode], oof_has), (axes[1], f"test ({N_FOLDS} averaged)", fin_pres, test_has)]:
    bins = np.linspace(0, 1, 21)
    ax.hist(prob[~h], bins=bins, alpha=0.7, color=C_EMPTY, label="empty frames", density=True)
    ax.hist(prob[h], bins=bins, alpha=0.7, color=C_NERVE, label="nerve frames", density=True)
    ax.axvline(final_settings["pres_thr"], color="k", lw=1, ls="--")
    ax.set_title(title); ax.set_xlabel("detection head probability")
axes[0].set_ylabel("density"); axes[0].legend(frameon=False)
fig.suptitle("P3  Does the head separate nerve from empty frames? Dashed line = chosen threshold.", fontsize=9, y=1.04)
save_fig(fig, "P3_head_probability.png")

# P4: threshold sweep for the head
sweep_rows = []
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
for ax, title, prob, h in [(axes[0], "out-of-fold", oof_pres[final_mode], oof_has), (axes[1], f"test ({N_FOLDS} averaged)", fin_pres, test_has)]:
    thr = np.round(np.arange(0.05, 0.96, 0.05), 2)
    sens = np.array([(prob[h] >= t).mean() for t in thr])
    spec = np.array([(prob[~h] < t).mean() for t in thr])
    for t, a, b in zip(thr, sens, spec):
        sweep_rows.append({"data": title, "threshold": t, "sensitivity": round(float(a), 3), "specificity": round(float(b), 3),
                           "balanced accuracy": round(float((a + b) / 2), 3)})
    ax.plot(thr, sens, color=C_NERVE, label="sensitivity (nerve found)")
    ax.plot(thr, spec, color=C_EMPTY, label="specificity (empty kept empty)")
    ax.plot(thr, (sens + spec) / 2, color="k", lw=1, label="balanced accuracy")
    ax.axvline(final_settings["pres_thr"], color=C_GREY, ls="--", lw=0.8)
    ax.set_title(title); ax.set_xlabel("head threshold")
axes[0].legend(frameon=False, fontsize=7)
fig.suptitle("P4  Trade-off when the head decides 'nerve / no nerve'.", fontsize=9, y=1.04)
save_fig(fig, "P4_threshold_sweep.png")
pd.DataFrame(sweep_rows).to_csv(os.path.join(TABLE_DIR, "B3_threshold_sweep.csv"), index=False)

# P5: reliability diagram
fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), sharey=True)
for ax, title, prob, h in [(axes[0], "out-of-fold", oof_pres[final_mode], oof_has), (axes[1], f"test ({N_FOLDS} averaged)", fin_pres, test_has)]:
    edges = np.linspace(0, 1, 11)
    which = np.clip(np.digitize(prob, edges) - 1, 0, 9)
    xs = [prob[which == b].mean() for b in range(10) if (which == b).any()]
    ys = [h[which == b].mean() for b in range(10) if (which == b).any()]
    ns = [(which == b).sum() for b in range(10) if (which == b).any()]
    ax.plot([0, 1], [0, 1], color=C_GREY, lw=0.8)
    ax.scatter(xs, ys, s=[20 + 300 * n / max(ns) for n in ns], color=C_NERVE, alpha=0.8)
    ax.plot(xs, ys, color=C_NERVE, lw=0.8)
    ax.set_title(f"{title}  (ECE {ece(prob, h.astype(float)):.2f})"); ax.set_xlabel("predicted probability")
axes[0].set_ylabel("share of frames with a nerve")
fig.suptitle("P5  Calibration of the head. On the diagonal = probabilities can be read as probabilities. Dot size = frames.", fontsize=9, y=1.04)
save_fig(fig, "P5_calibration.png")

# P6: configurations compared (test rows) with intervals
test_rows = config_df[config_df.dataset == "test (10 pt)"].reset_index(drop=True)
labels = [f"{r.models} | {r.flip} | {r.settings}" for r in test_rows.itertuples()]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, col, title in [(axes[0], "balanced", "balanced score"), (axes[1], "nerve_dice", "Dice on nerve frames")]:
    y = np.arange(len(test_rows))[::-1]
    err = [test_rows[col] - test_rows[col + "_lo"], test_rows[col + "_hi"] - test_rows[col]]
    ax.errorbar(test_rows[col], y, xerr=err, fmt="o", color=C_NERVE, ecolor=C_GREY, capsize=2)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7); ax.set_title(title)
    if col == "balanced":
        ax.axvline(0.5, color=C_RED, ls="--", lw=0.8); ax.text(0.5, y.max() + 0.6, " always empty", color=C_RED, fontsize=7)
fig.suptitle("P6  Test patients: every way of using the models, with 95% intervals over patients.", fontsize=9, y=1.04)
save_fig(fig, "P6_configurations.png")

# P7: Dice against nerve size (test nerve frames, final configuration)
area = test_true[test_has].sum(axis=(1, 2))
dice_nerve = fin_dice[test_has]
fig, ax = plt.subplots(figsize=(5.5, 3.8))
ax.scatter(area, dice_nerve, s=8, alpha=0.4, color=C_NERVE)
edges = np.quantile(area, np.linspace(0, 1, 6))
mid, med = [], []
for a, b in zip(edges[:-1], edges[1:]):
    sel = (area >= a) & (area <= b)
    if sel.any():
        mid.append(np.median(area[sel])); med.append(np.median(dice_nerve[sel]))
ax.plot(mid, med, "o-", color=C_RED, label="median per size quintile")
ax.set_xlabel(f"true nerve area (pixels on the {H}x{W} grid)"); ax.set_ylabel("Dice"); ax.legend(frameon=False)
ax.set_title("P7  Is a small nerve harder to find?", fontsize=9)
save_fig(fig, "P7_dice_vs_size.png")

# P8: per fold and per patient
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
fold_vals = []
for k in range(N_FOLDS):
    sel = np.where(fold_of_dev == k)[0]
    h = oof_has[sel]
    fold_vals.append(balanced_score(frame_dice(oof_final_masks[sel], oof_true[sel]), h) if h.any() and not h.all() else np.nan)
axes[0].bar(range(N_FOLDS), fold_vals, color=C_NERVE)
axes[0].axhline(0.5, color=C_RED, ls="--", lw=0.8); axes[0].set_xlabel("fold"); axes[0].set_ylabel("balanced score")
axes[0].set_title("out-of-fold, per fold")
pats = np.unique(test_pat)
nd = [fin_dice[(test_pat == p) & test_has].mean() if ((test_pat == p) & test_has).any() else np.nan for p in pats]
fd = [fin_found[(test_pat == p) & test_has].mean() if ((test_pat == p) & test_has).any() else np.nan for p in pats]
x = np.arange(len(pats))
axes[1].bar(x - 0.2, nd, 0.4, color=C_NERVE, label="Dice on nerve frames")
axes[1].bar(x + 0.2, fd, 0.4, color=C_EMPTY, label="share of nerve frames found")
axes[1].set_xticks(x); axes[1].set_xticklabels([str(p) for p in pats]); axes[1].set_xlabel("test patient"); axes[1].legend(frameon=False, fontsize=7)
axes[1].set_title("test, per patient")
fig.suptitle("P8  Spread between folds and between patients.", fontsize=9, y=1.04)
save_fig(fig, "P8_folds_and_patients.png")

# P9: why flip averaging and averaging 5 models can hurt
peak_sets = {
    "1 model\nno flip": np.concatenate([peak(s[test_has]) for s, _ in test_preds["no flip"]]),
    "1 model\nflip": np.concatenate([peak(s[test_has]) for s, _ in test_preds["flip"]]),
    f"{N_FOLDS} avg\nno flip": peak(average_models(test_preds["no flip"])[0][test_has]),
    f"{N_FOLDS} avg\nflip": peak(average_models(test_preds["flip"])[0][test_has]),
}
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
axes[0].boxplot(list(peak_sets.values()), showfliers=False)
axes[0].set_xticks(range(1, len(peak_sets) + 1))
axes[0].set_xticklabels(list(peak_sets.keys()), fontsize=7)
axes[0].axhline(0.5, color=C_RED, ls="--", lw=0.8)
axes[0].set_ylabel("highest pixel probability"); axes[0].set_title("how confident is the model on nerve frames?")
valid = pair_dice[np.isfinite(pair_dice)]
axes[1].hist(valid, bins=np.linspace(0, 1, 21), color=C_NERVE)
axes[1].set_xlabel("Dice between mask of the image and mask of its mirror image")
axes[1].set_ylabel("test nerve frames"); axes[1].set_title(f"flip consistency (median {np.median(valid):.2f})")
fig.suptitle("P9  Averaging with a mirrored prediction only helps if the model gives the same answer for both. Red line = 0.5.", fontsize=9, y=1.04)
save_fig(fig, "P9_flip_and_ensemble.png")

# P10: examples - worst, median, best nerve frames and false alarms (final configuration, test patients)
nerve_pos = np.where(test_has)[0]
order = nerve_pos[np.argsort(fin_dice[nerve_pos])]
mid_start = max(0, len(order) // 2 - 2)
fp_pos = np.where(~test_has & fin_found)[0]
groups_to_show = [("worst", order[:4]), ("median", order[mid_start:mid_start + 4]), ("best", order[-4:]),
                  ("false alarm", fp_pos[rng.permutation(len(fp_pos))[:4]])]
groups_to_show = [(name, ids) for name, ids in groups_to_show if len(ids) > 0]     # no empty rows
fig, axes = plt.subplots(len(groups_to_show), 4, figsize=(9, 2.3 * len(groups_to_show)), squeeze=False)
for r, (name, ids) in enumerate(groups_to_show):
    for c in range(4):
        ax = axes[r, c]
        ax.axis("off")
        if c < len(ids):
            i = ids[c]
            ax.imshow(X[test_idx[i]], cmap="gray", vmin=0, vmax=1)
            if test_true[i].any():
                ax.contour(test_true[i].astype(float), levels=[0.5], colors="#2ecc71", linewidths=1.2)
            if fin_masks[i].any():
                ax.contour(fin_masks[i].astype(float), levels=[0.5], colors="#e74c3c", linewidths=1.2)
            ax.set_title(f"{name}: Dice {fin_dice[i]:.2f}, head {fin_pres[i]:.2f}", fontsize=7)
fig.suptitle("P10  Test frames. Green = true nerve, red = prediction. Rows: worst, median, best, false alarms.", fontsize=9, y=1.04)
save_fig(fig, "P10_examples.png")


##### 10. Report, config for C++, MLflow

with open(os.path.join(TABLE_DIR, "report.md"), "w") as f:
    f.write("# Diagnostics report (step 9)\n\n")
    f.write(f"Final configuration, chosen on out-of-fold predictions only: flip averaging = {final_mode}, "
            f"thresholds = {final_kind} {final_settings}\n\n")
    for title, note, df in report_parts:
        f.write(f"## {title}\n\n")
        if note:
            f.write(note.replace("\n", "  \n") + "\n\n")
        f.write(md_table(df) + "\n\n")

post = []
if final_config["flip_average"]:
    post += ["apply sigmoid to both outputs", "do the same for the mirrored input (flip left-right), flip the segmentation back, average the two"]
else:
    post += ["apply sigmoid to both outputs (no mirrored input)"]
post += [f"average over all {N_FOLDS} models",
         f"if the presence probability is below {final_settings['pres_thr']}: empty mask",
         f"set pixels with probability above {final_settings['pix_thr']} to nerve"]
if final_largest:
    post += ["keep only the largest connected blob"]
post += [f"if the blob has fewer than {final_settings['min_area']} pixels: empty mask"]
cpp_config = {
    "models": [f"nerve_unet_fold{k}_torchscript.pt" for k in range(N_FOLDS)],
    "input_shape": [1, 1, H, W],
    "preprocessing": ["load the image as grayscale", f"resize to {W}x{H} (width x height) with bilinear interpolation (PIL Image.BILINEAR)",
                      "divide by 255, so values are float32 between 0 and 1"],
    "model_outputs": [f"segmentation logits, shape (1, 1, {H}, {W})", "presence logit, shape (1)"],
    "postprocessing": post,
    "thresholds": final_settings,
    "chosen_on": "out-of-fold predictions of the non-test patients only",
    "note": "Run models/nerve_reference_input.bin through libtorch and compare with the reference outputs before trusting the export."}
with open(os.path.join(OUT_DIR, "models", "nerve_unet_config.json"), "w") as f:
    json.dump(cpp_config, f, indent=2)

final_row = config_df[(config_df.dataset == "test (10 pt)") & (config_df.models == f"{N_FOLDS} averaged")
                      & (config_df.flip == final_mode) & (config_df.settings == ("default" if final_kind == "default" else "tuned"))].iloc[0]
for name in ["balanced", "nerve_dice", "found", "empty_kept"]:
    mlflow.log_metric(f"test_{name}", float(final_row[name]))
    mlflow.log_metric(f"test_{name}_ci_low", float(final_row[name + "_lo"]))
    mlflow.log_metric(f"test_{name}_ci_high", float(final_row[name + "_hi"]))
mlflow.log_params({"final_flip": final_mode, "final_thresholds": final_kind, **{f"final_{k}": v for k, v in final_settings.items()}})
mlflow.log_artifacts(TABLE_DIR, artifact_path="tables")
mlflow.log_artifacts(PLOT_DIR, artifact_path="plots")
mlflow.log_artifact(os.path.join(OUT_DIR, "final_config.json"))
try:
    mlflow.log_artifact(__file__)
except NameError:
    pass
mlflow.end_run()
print(f"\nsaved {len(saved_plots)} plots to {PLOT_DIR} and tables to {TABLE_DIR} (report.md has all tables)")
print("next: run step10_explain.py")



######## Output 

'''
dev: 4436 frames, 37 patients   test: 1199 frames, 10 patients
  model 1/5 predicted
  model 2/5 predicted
  model 3/5 predicted
  model 4/5 predicted
  model 5/5 predicted
predictions took 37 s
/tmp/ipykernel_58/1992115892.py:164: RuntimeWarning: Mean of empty slice
  pair_dice = np.nanmean(np.stack(pair_dice), axis=0)       # per test nerve frame, mean over models
tuned thresholds (no flip): {'pres_thr': 0.6, 'pix_thr': 0.6, 'min_area': 200}
tuned thresholds (flip): {'pres_thr': 0.5, 'pix_thr': 0.4, 'min_area': 200}
threshold search took 105 s

=== Table A: configurations (value [95% interval over patients]) ===
balanced score = half the Dice on nerve frames + half the Dice on empty frames. 'Always empty' scores 0.5.
OOF rows: every frame is predicted by a model that never saw its patient.
Test rows: the test patients were not used for training or for choosing anything.
     dataset          models    flip                        settings       balanced score     nerve-frame Dice          nerve found     empty kept empty
 OOF (37 pt)               -       -                    always empty 0.500 [0.500, 0.500] 0.000 [0.000, 0.000] 0.000 [0.000, 0.000] 1.000 [1.000, 1.000]
 OOF (37 pt) 1 (out-of-fold) no flip                         default 0.566 [0.518, 0.610] 0.337 [0.238, 0.425] 0.510 [0.387, 0.618] 0.796 [0.722, 0.863]
 OOF (37 pt) 1 (out-of-fold) no flip tuned, same frames (optimistic) 0.585 [0.544, 0.625] 0.242 [0.152, 0.327] 0.320 [0.206, 0.425] 0.927 [0.893, 0.957]
 OOF (37 pt) 1 (out-of-fold) no flip             tuned, cross-fitted 0.581 [0.540, 0.621] 0.264 [0.171, 0.351] 0.354 [0.235, 0.467] 0.898 [0.846, 0.942]
 OOF (37 pt) 1 (out-of-fold)    flip                         default 0.511 [0.487, 0.540] 0.104 [0.053, 0.159] 0.233 [0.126, 0.339] 0.918 [0.877, 0.954]
 OOF (37 pt) 1 (out-of-fold)    flip tuned, same frames (optimistic) 0.567 [0.526, 0.610] 0.216 [0.124, 0.306] 0.300 [0.174, 0.420] 0.918 [0.881, 0.953]
 OOF (37 pt) 1 (out-of-fold)    flip             tuned, cross-fitted 0.521 [0.498, 0.546] 0.104 [0.051, 0.165] 0.147 [0.074, 0.232] 0.938 [0.901, 0.969]
test (10 pt)               -       -                    always empty 0.500 [0.500, 0.500] 0.000 [0.000, 0.000] 0.000 [0.000, 0.000] 1.000 [1.000, 1.000]
test (10 pt)   1 (mean of 5) no flip                         default 0.565 [0.512, 0.613] 0.319 [0.183, 0.449] 0.480 [0.287, 0.653] 0.811 [0.721, 0.876]
test (10 pt)      5 averaged no flip                         default 0.620 [0.545, 0.680] 0.334 [0.151, 0.511] 0.435 [0.201, 0.654] 0.905 [0.814, 0.962]
test (10 pt)   1 (mean of 5) no flip                           tuned 0.573 [0.528, 0.611] 0.216 [0.099, 0.328] 0.298 [0.141, 0.441] 0.929 [0.874, 0.965]
test (10 pt)      5 averaged no flip                           tuned 0.593 [0.534, 0.645] 0.221 [0.078, 0.356] 0.280 [0.103, 0.452] 0.965 [0.920, 0.992]
test (10 pt)   1 (mean of 5)    flip                         default 0.500 [0.480, 0.519] 0.107 [0.065, 0.146] 0.216 [0.139, 0.283] 0.894 [0.853, 0.927]
test (10 pt)      5 averaged    flip                         default 0.556 [0.521, 0.582] 0.146 [0.061, 0.222] 0.234 [0.094, 0.372] 0.966 [0.922, 0.991]
test (10 pt)   1 (mean of 5)    flip                           tuned 0.547 [0.510, 0.580] 0.199 [0.110, 0.285] 0.288 [0.166, 0.402] 0.896 [0.845, 0.936]
test (10 pt)      5 averaged    flip                           tuned 0.576 [0.531, 0.616] 0.173 [0.066, 0.279] 0.226 [0.083, 0.370] 0.978 [0.945, 0.997]

chosen on OOF only: flip = no flip, thresholds = tuned {'pres_thr': 0.6, 'pix_thr': 0.6, 'min_area': 200} (OOF balanced 0.581)

=== Table B: frame-level detection of a nerve ===
AUROC and AP do not depend on a threshold. 'max pixel probability' uses the segmentation output instead of the head.
If the head is not clearly better than the max pixel probability, the extra head adds little.
                                    scores                AUROC                   AP prevalence sens/spec @0.50 sens/spec @0.60
              OOF, detection head, no flip 0.684 [0.615, 0.745] 0.590 [0.450, 0.702]       0.39     0.54 / 0.74     0.37 / 0.86
       OOF, max pixel probability, no flip 0.723 [0.650, 0.775] 0.653 [0.515, 0.753]       0.39     0.76 / 0.51               -
                 OOF, detection head, flip 0.654 [0.579, 0.724] 0.529 [0.382, 0.662]       0.39     0.36 / 0.84     0.14 / 0.94
          OOF, max pixel probability, flip 0.677 [0.605, 0.737] 0.547 [0.428, 0.660]       0.39     0.51 / 0.72               -
           test, head, 5 averaged, no flip 0.748 [0.653, 0.828] 0.754 [0.632, 0.837]       0.51     0.44 / 0.88     0.29 / 0.95
       test, head, 1 model (mean), no flip 0.700 [0.607, 0.772] 0.705 [0.577, 0.790]       0.51     0.49 / 0.78     0.32 / 0.90
test, max pixel prob., 5 averaged, no flip 0.813 [0.744, 0.856] 0.796 [0.687, 0.850]       0.51     0.79 / 0.68               -
              test, head, 5 averaged, flip 0.737 [0.635, 0.819] 0.726 [0.613, 0.815]       0.51     0.25 / 0.94     0.02 / 0.99
          test, head, 1 model (mean), flip 0.684 [0.593, 0.754] 0.674 [0.552, 0.762]       0.51     0.32 / 0.85     0.09 / 0.97
   test, max pixel prob., 5 averaged, flip 0.798 [0.740, 0.840] 0.778 [0.681, 0.838]       0.51     0.54 / 0.85               -

=== Table B2: calibration of the detection head (no flip) ===
ECE = average gap between predicted probability and the real share of nerve frames (0 is perfect).
             data ECE (10 bins) mean prob. on nerve frames mean prob. on empty frames
              OOF         0.066                       0.52                       0.39
test (5 averaged)         0.085                       0.49                       0.36

=== Table C: segmentation quality on nerve frames only (no gating, pixel threshold 0.5) ===
Dice/IoU measure the overlap. Precision = share of predicted pixels that are nerve. Recall = share of nerve pixels found.
                  predictions                 dice                  iou            precision               recall
        OOF, 1 model, no flip 0.443 [0.349, 0.524] 0.349 [0.270, 0.418] 0.522 [0.426, 0.603] 0.453 [0.349, 0.543]
           OOF, 1 model, flip 0.174 [0.107, 0.247] 0.126 [0.075, 0.182] 0.305 [0.205, 0.407] 0.146 [0.087, 0.212]
test, 1 model (mean), no flip 0.472 [0.368, 0.565] 0.366 [0.274, 0.447] 0.525 [0.434, 0.603] 0.514 [0.399, 0.620]
    test, 5 averaged, no flip 0.500 [0.375, 0.615] 0.400 [0.285, 0.505] 0.622 [0.525, 0.696] 0.477 [0.334, 0.611]
   test, 1 model (mean), flip 0.183 [0.144, 0.217] 0.135 [0.104, 0.165] 0.278 [0.228, 0.320] 0.169 [0.130, 0.202]
       test, 5 averaged, flip 0.278 [0.159, 0.395] 0.211 [0.118, 0.303] 0.475 [0.309, 0.635] 0.225 [0.125, 0.320]

=== Table D1: out-of-fold results per fold (final configuration: no flip, tuned) ===
Large differences between folds mean the result depends on which patients are in the training data.
Thresholds here were tuned on all folds, so these values are slightly optimistic.
 fold  patients  frames nerve share balanced nerve Dice nerve found empty kept head AUROC
    0         7     840        0.34    0.533      0.204        0.31       0.86      0.604
    1         7     840        0.29    0.505      0.022        0.03       0.99      0.496
    2         7     840        0.22    0.515      0.097        0.14       0.93      0.588
    3         8     958        0.59    0.611      0.326        0.42       0.90      0.766
    4         8     958        0.44    0.645      0.347        0.44       0.94      0.799

=== Table D2: test results per patient (final configuration, models averaged) ===
 patient  frames nerve share nerve Dice nerve found empty kept head AUROC
       5     120        0.27      0.000        0.00       1.00      0.779
      11     120        0.61      0.582        0.77       0.81      0.788
      12     120        0.17      0.000        0.00       1.00      0.488
      19     120        0.57      0.012        0.01       1.00      0.544
      23     120        0.78      0.086        0.11       1.00      0.718
      29     120        0.51      0.167        0.25       0.97      0.787
      31     120        0.33      0.000        0.00       0.99      0.694
      32     120        0.81      0.352        0.43       0.78      0.673
      34     119        0.62      0.518        0.62       0.91      0.899
      43     120        0.39      0.000        0.00       1.00      0.769

saved 10 plots to outputs/plots and tables to outputs/tables (report.md has all tables)
next: run step10_explain.py

'''
