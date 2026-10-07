##### Step 10: explainability (what does the model look at?)
# Reads the models, the split and final_config.json from steps 8 and 9. Trains nothing.
#
# Four ways to look at the model, all for the question "why does it say nerve / where is the nerve?":
#   Grad-CAM        gradient of the detection-head output with respect to the deepest layer (quarter size)
#   Head-CAM        the same gradient multiplied with the layer, without averaging. The head is linear
#                   (mean + max of the layer -> 1 number), so this splits the head output EXACTLY into
#                   one contribution per location. Check: the contributions add up to the output.
#   Seg-Grad-CAM    like Grad-CAM, but for the segmentation: gradient of the summed mask logits
#                   (inside the predicted mask) with respect to the last decoder layer (full size).
#                   The segmentation also uses skip connections that bypass the deepest layer,
#                   so the deepest layer alone would not explain it.
#   Occlusion       model-agnostic: hide a 16x16 patch, measure how much the head output drops
# A picture of a map proves nothing. So the maps are tested with numbers:
#   pointing game   does the strongest point of the map lie on the true nerve (3 pixel tolerance)?
#   energy in nerve how much of the map lies on the true nerve, divided by the nerve's share of the image
#                   (1 = no better than spreading the map evenly)
#   deletion        remove the most important pixels first: does the detection drop faster than when
#                   random pixels are removed? (faithfulness)
#   sanity check    randomize the weights layer by layer: the maps must change. If they do not, the
#                   map describes the image and not the model (Adebayo et al., 2018).
#   edge reliance   how much of the map lies in the outer 10% of the image (probe contact, borders)?
# Explanations are for one forward pass of one fold model, without flip averaging.
# Maps are shown on the test patients, which were not used for training or for choosing anything.
#
# Output (in OUT_DIR): tables/E*.csv, tables/explain_report.md, plots/X*.png
# Needs:  pip install mlflow torch scikit-learn scipy pandas matplotlib

import json
import os
import time
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)       # all-NaN slices in the interval calculation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from scipy.stats import spearmanr

OUT_DIR = os.environ.get("OUT_DIR", "outputs")
H, W = 96, 128
SEED = 0
N_EXPLAIN = int(os.environ.get("N_EXPLAIN", 150))     # test nerve frames used for occlusion, deletion, the main table
N_SANITY = int(os.environ.get("N_SANITY", 40))        # frames for the weight-randomization check
N_BOOT = int(os.environ.get("N_BOOT", 2000))
OCC, STRIDE = 16, 8                                   # occlusion patch and step, in pixels
TOLERANCE = 3                                         # pointing game: pixels around the true nerve that still count
BORDER = 0.10                                         # outer share of the image counted as "edge"
FRACTIONS = [0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]  # share of pixels removed in the deletion test
RUN_NAME = "step10_explain"

TABLE_DIR = os.path.join(OUT_DIR, "tables")
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
rng = np.random.default_rng(SEED)

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 130, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
C_NERVE, C_EMPTY, C_GREY, C_RED = "#1b6ca8", "#d9822b", "#8a8a8a", "#c0392b"
METHODS = ["Grad-CAM", "Head-CAM", "Seg-Grad-CAM", "Occlusion"]
METHOD_COLORS = {"Grad-CAM": "#1b6ca8", "Head-CAM": "#2a9d8f", "Seg-Grad-CAM": "#8e44ad", "Occlusion": "#d9822b", "Random": C_GREY}

mlflow.set_tracking_uri(f"sqlite:///{os.path.join(OUT_DIR, 'mlflow.db')}")
mlflow.set_experiment("nerve_segmentation")
mlflow.start_run(run_name=RUN_NAME)
mlflow.log_params({"step": 10, "n_explain": N_EXPLAIN, "n_sanity": N_SANITY, "occlusion_patch": OCC,
                   "occlusion_stride": STRIDE, "pointing_tolerance_px": TOLERANCE})


##### 1. Load the cache, the split, the final configuration and the models

cache = np.load(os.path.join(OUT_DIR, "data_cache.npz"))
X = cache["X"].astype(np.float32) / 255.0
Y = cache["Y"].astype(np.float32)
patients = cache["patients"]
has_nerve = Y.sum(axis=(1, 2)) > 0
split = np.load(os.path.join(OUT_DIR, "split.npz"))
test_idx, fold_of_dev = split["test_idx"], split["fold_of_dev"]
N_FOLDS = int(fold_of_dev.max()) + 1
test_true, test_has, test_pat = Y[test_idx] > 0.5, has_nerve[test_idx], patients[test_idx]
with open(os.path.join(OUT_DIR, "final_config.json")) as f:
    final_config = json.load(f)
PRES_THR = final_config["settings"]["pres_thr"]
print(f"test: {len(test_idx)} frames, {len(set(test_pat))} patients; detection threshold {PRES_THR}")

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
model0 = models[0]                      # the model used for occlusion, deletion, sanity check and the gallery


##### 2. The four explanation methods

def gradient_maps(model, ids, signed=False):
    """Grad-CAM, Head-CAM and Seg-Grad-CAM for the frames X[ids] (ids index into the full arrays).
    Returns maps of shape (n, H, W), plus the detection probability and the Head-CAM check.
    signed=False: negative values are cut to 0 (only evidence FOR a nerve), the normal setting.
    signed=True: the absolute value of the signed map is returned, used in the weight-randomization check."""
    out = {"Grad-CAM": [], "Head-CAM": [], "Seg-Grad-CAM": []}
    probs, completeness = [], []
    for start in range(0, len(ids), 64):
        xb = torch.from_numpy(X[ids[start:start + 64]]).unsqueeze(1).to(device)
        kept = {}
        hook = model.dec1.register_forward_hook(lambda module, inputs, output: kept.update(d1=output))
        seg_logits, pres_logit, m = model(xb, return_features=True)      # m: deepest layer, (B, 64, H/4, W/4)
        hook.remove()
        d1 = kept["d1"]                                                  # last decoder layer, (B, 16, H, W)

        # detection head: Grad-CAM (gradient averaged over space) and Head-CAM (gradient times layer, kept per location)
        g = torch.autograd.grad(pres_logit.sum(), m, retain_graph=True)[0]
        grad_cam = (g.mean(dim=(2, 3), keepdim=True) * m).sum(dim=1, keepdim=True)
        head_cam = (g * m).sum(dim=1, keepdim=True)
        # check: the contributions must add up to the head output minus its bias
        completeness.append((head_cam.sum(dim=(1, 2, 3)) - (pres_logit - model.presence.bias)).abs().detach().cpu().numpy())

        # segmentation: sum of mask logits inside the predicted mask (if empty: inside the model's best guess)
        prob = torch.sigmoid(seg_logits)
        region = prob > 0.5
        nothing = ~region.flatten(1).any(dim=1)
        best_guess = prob >= 0.8 * prob.amax(dim=(1, 2, 3), keepdim=True)
        region = torch.where(nothing[:, None, None, None], best_guess, region).float()
        g2 = torch.autograd.grad((seg_logits * region).sum(), d1)[0]
        seg_cam = (g2.mean(dim=(2, 3), keepdim=True) * d1).sum(dim=1, keepdim=True)

        for name, cam in [("Grad-CAM", grad_cam), ("Head-CAM", head_cam), ("Seg-Grad-CAM", seg_cam)]:
            if signed:
                up = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)[:, 0].abs()
            else:
                up = F.interpolate(F.relu(cam), size=(H, W), mode="bilinear", align_corners=False)[:, 0]
            out[name].append(up.detach().cpu().numpy())
        probs.append(torch.sigmoid(pres_logit).detach().cpu().numpy())
    return ({k: np.concatenate(v) for k, v in out.items()}, np.concatenate(probs), float(np.concatenate(completeness).max()))

def occlusion_map(model, image):
    """Drop of the detection logit when a OCC x OCC patch is replaced by the frame mean. image: (H, W)."""
    positions = [(y, x) for y in range(0, H - OCC + 1, STRIDE) for x in range(0, W - OCC + 1, STRIDE)]
    batch = np.repeat(image[None, None], len(positions) + 1, axis=0)          # last entry stays unchanged
    for i, (y, x) in enumerate(positions):
        batch[i, 0, y:y + OCC, x:x + OCC] = image.mean()
    with torch.no_grad():
        logits = model(torch.from_numpy(batch).to(device))[1].cpu().numpy()
    drop = logits[-1] - logits[:-1]
    total, count = np.zeros((H, W)), np.zeros((H, W))
    for (y, x), d in zip(positions, drop):
        total[y:y + OCC, x:x + OCC] += d
        count[y:y + OCC, x:x + OCC] += 1
    return np.maximum(total / np.maximum(count, 1), 0).astype(np.float32)   # only "this region supports a nerve"

def presence_prob(model, images):
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(images).unsqueeze(1).to(device))[1]).cpu().numpy()


##### 3. Maps for the test nerve frames

nerve_pos = np.where(test_has)[0]                                           # positions inside the test arrays
subset = np.sort(rng.choice(nerve_pos, size=min(N_EXPLAIN, len(nerve_pos)), replace=False))
t0 = time.time()
maps, probs0, completeness_error = gradient_maps(model0, test_idx[subset])
maps["Occlusion"] = np.stack([occlusion_map(model0, X[test_idx[i]]) for i in subset])
maps["Random"] = rng.random((len(subset), H, W)).astype(np.float32)        # baseline: a map with no information
print(f"maps for {len(subset)} test nerve frames took {time.time() - t0:.0f} s")
print(f"Head-CAM check: largest |sum of contributions - (output - bias)| = {completeness_error:.2e}")
detected = probs0 >= PRES_THR                                               # frames where the model says "nerve"
sub_true, sub_pat = test_true[subset], test_pat[subset]
print(f"{detected.sum()} of {len(subset)} frames are detected by model 1 (threshold {PRES_THR})")


##### 4. Numbers for the maps

border_mask = np.zeros((H, W), dtype=bool)
bh, bw = int(round(BORDER * H)), int(round(BORDER * W))
border_mask[:bh], border_mask[-bh:], border_mask[:, :bw], border_mask[:, -bw:] = True, True, True, True

def map_scores(map_array, true):
    """Per-frame scores of a map (n, H, W) against the true nerve masks."""
    n = len(map_array)
    hit, energy, enrich, chance, edge, empty = (np.full(n, np.nan) for _ in range(6))
    for i in range(n):
        mp = map_array[i]
        grown = ndimage.binary_dilation(true[i], iterations=TOLERANCE)
        chance[i] = grown.mean()
        total = mp.sum()
        empty[i] = float(mp.max() <= 0)
        hit[i] = float(mp.max() > 0 and grown[np.unravel_index(np.argmax(mp), mp.shape)])
        if total > 0:
            energy[i] = mp[true[i]].sum() / total
            enrich[i] = energy[i] / true[i].mean()
            edge[i] = (mp[border_mask].sum() / total) / border_mask.mean()
    return {"hit": hit, "energy": energy, "enrichment": enrich, "chance": chance, "edge": edge, "empty": empty}

def boot_mean(values, groups, n_boot=N_BOOT):
    """Mean with a 95% interval; whole patients are redrawn."""
    draw_rng = np.random.default_rng(SEED)
    ids = np.unique(groups)
    members = [np.where(groups == g)[0] for g in ids]
    boots = []
    for _ in range(n_boot):
        drawn = draw_rng.integers(0, len(ids), len(ids))
        idx = np.concatenate([members[j] for j in drawn])
        boots.append(np.nanmean(values[idx]) if np.isfinite(values[idx]).any() else np.nan)
    return float(np.nanmean(values)), float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))

def fmt(m, lo, hi, digits=2):
    return f"{m:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"

def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)

report_parts = []
def add_table(name, title, note, df):
    df.to_csv(os.path.join(TABLE_DIR, name + ".csv"), index=False)
    report_parts.append((title, note, df))
    print(f"\n=== {title} ===")
    if note:
        print(note)
    print(df.to_string(index=False))

# Table E1: pointing game, energy in nerve, edge reliance (model 1, the sampled test nerve frames)
e1_rows, scores_by_method = [], {}
for method in METHODS + ["Random"]:
    scores_by_method[method] = map_scores(maps[method], sub_true)
for subset_name, keep in [("all sampled nerve frames", np.ones(len(subset), dtype=bool)), ("frames the model detects", detected)]:
    for method in METHODS + ["Random"]:
        s = scores_by_method[method]
        if keep.sum() == 0:
            continue
        row = {"method": method, "frames": subset_name, "n": int(keep.sum()),
               "pointing hit": fmt(*boot_mean(s["hit"][keep], sub_pat[keep])),
               "chance": f"{np.nanmean(s['chance'][keep]):.2f}",
               "energy in nerve": fmt(*boot_mean(s["energy"][keep], sub_pat[keep])),
               "energy / nerve share": fmt(*boot_mean(s["enrichment"][keep], sub_pat[keep]), digits=1),
               "edge reliance": f"{np.nanmean(s['edge'][keep]):.2f}",
               "empty maps": f"{np.mean(s['empty'][keep]):.2f}"}
        e1_rows.append(row)
add_table("E1_map_quality", "Table E1: do the maps point at the nerve? (model 1, test patients, value [95% interval over patients])",
          "pointing hit: strongest point lies within 3 pixels of the true nerve. chance: hit rate of a random point.\n"
          "energy / nerve share: 1 = map is spread evenly, higher = map is concentrated on the nerve.\n"
          "edge reliance: map share in the outer 10% of the image divided by the area share (1 = neutral, above 1 = relies on the edges).\n"
          "Random = a map with no information, shown as a reference.", pd.DataFrame(e1_rows))

# Table E2: the same for the 3 gradient methods on ALL test nerve frames, one row per model (stability across models)
e2_rows = []
for k, model in enumerate(models):
    mk, pk, _ = gradient_maps(model, test_idx[nerve_pos])
    for method in ["Grad-CAM", "Head-CAM", "Seg-Grad-CAM"]:
        s = map_scores(mk[method], test_true[nerve_pos])
        e2_rows.append({"model": k + 1, "method": method, "frames": len(nerve_pos), "detected": f"{(pk >= PRES_THR).mean():.2f}",
                        "pointing hit": f"{np.nanmean(s['hit']):.3f}", "energy / nerve share": f"{np.nanmean(s['enrichment']):.1f}"})
e2_df = pd.DataFrame(e2_rows)
spread = e2_df.assign(hit=e2_df["pointing hit"].astype(float)).groupby("method").hit.agg(["mean", "std"]).reset_index()
add_table("E2_model_stability", "Table E2: the same test per fold model (all test nerve frames)",
          "If the pointing hit differs a lot between models, the explanation depends on which patients the model was trained on.", e2_df)
print("pointing hit, mean +- sd over the models:")
print(spread.to_string(index=False))

# Table E3: deletion test (faithfulness)
def delete_curve(ranking, ids_local):
    """Detection probability of model 1 after removing the top-ranked pixels (replaced by the frame mean)."""
    curves = np.zeros((len(ids_local), len(FRACTIONS)))
    for j, i in enumerate(ids_local):
        image = X[test_idx[subset[i]]]
        order = np.argsort(-(ranking[i].ravel() + 1e-9 * rng.random(H * W)))        # tiny noise breaks ties
        batch = np.repeat(image[None], len(FRACTIONS), axis=0)
        for f_i, frac in enumerate(FRACTIONS):
            kill = order[:int(round(frac * H * W))]
            flat = batch[f_i].reshape(-1)
            flat[kill] = image.mean()
        curves[j] = presence_prob(model0, batch)
    return curves

del_ids = np.where(detected)[0]
deletion = {}
if len(del_ids) > 0:
    for method in METHODS:
        deletion[method] = delete_curve(maps[method], del_ids)
    deletion["Random"] = np.mean([delete_curve(rng.random((len(subset), H, W)), del_ids) for _ in range(3)], axis=0)
    area = {m: np.trapezoid(c, FRACTIONS, axis=1) / FRACTIONS[-1] for m, c in deletion.items()}   # mean detection prob. over the curve
    del_rows = []
    for method in METHODS + ["Random"]:
        row = {"method": method, "n": len(del_ids), "detection after removing 5%": f"{deletion[method][:, 2].mean():.2f}",
               "after 20%": f"{deletion[method][:, 4].mean():.2f}", "mean over the curve (lower = better)": fmt(*boot_mean(area[method], sub_pat[del_ids]), digits=3)}
        if method != "Random":
            row["difference to random"] = fmt(*boot_mean(area[method] - area["Random"], sub_pat[del_ids]), digits=3)
        else:
            row["difference to random"] = "-"
        del_rows.append(row)
    add_table("E3_deletion", "Table E3: deletion test (model 1, frames it detects)",
              "The most important pixels are removed first. A faithful map makes the detection probability fall faster than random removal.\n"
              "A difference to random below 0 with an interval that stays below 0 = better than random.", pd.DataFrame(del_rows))
else:
    print("no detected frames, deletion test skipped")

# Table E4: sanity check, randomize the weights from the output back to the input
def randomized_copy(model, upto):
    """Copy of the model with fresh random weights in the first `upto` stages, counted from the output."""
    stages = [("heads", [model.out, model.presence]), ("dec1", [model.up1, model.dec1]), ("dec2", [model.up2, model.dec2]),
              ("middle", [model.middle]), ("enc2", [model.enc2]), ("enc1", [model.enc1])]
    clone = UNet().to(device)
    clone.load_state_dict(model.state_dict())
    torch.manual_seed(SEED + 100)
    parts = {"heads": [clone.out, clone.presence], "dec1": [clone.up1, clone.dec1], "dec2": [clone.up2, clone.dec2],
             "middle": [clone.middle], "enc2": [clone.enc2], "enc1": [clone.enc1]}
    for name, _ in stages[:upto]:
        for part in parts[name]:
            for sub in part.modules():
                if hasattr(sub, "reset_parameters"):
                    sub.reset_parameters()
    return clone.eval()

sanity_local = np.where(detected)[0][:N_SANITY]
levels = ["original", "heads", "+ dec1", "+ dec2", "+ middle", "+ enc2", "+ enc1 (all)"]
sanity_rows, sanity_curves = [], {m: [] for m in ["Grad-CAM", "Head-CAM", "Seg-Grad-CAM"]}
if len(sanity_local) > 0:
    ids_full = test_idx[subset[sanity_local]]
    reference, _, _ = gradient_maps(model0, ids_full, signed=True)
    for level in range(0, 7):
        if level == 0:
            row = {"randomized layers": levels[0]}
            for m in sanity_curves:
                row[m] = "1.00"
                sanity_curves[m].append((1.0, 0.0))
        else:
            clone = randomized_copy(model0, level)
            rand_maps, _, _ = gradient_maps(clone, ids_full, signed=True)
            row = {"randomized layers": levels[level]}
            for m in sanity_curves:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    rho = np.array([spearmanr(reference[m][i].ravel(), rand_maps[m][i].ravel()).correlation for i in range(len(ids_full))])
                ok = np.isfinite(rho)
                mean_rho, sd_rho = (float(rho[ok].mean()), float(rho[ok].std())) if ok.any() else (np.nan, np.nan)
                row[m] = f"{mean_rho:.2f} +- {sd_rho:.2f}" if ok.any() else "map is empty"
                sanity_curves[m].append((mean_rho, sd_rho))
        sanity_rows.append(row)
    add_table("E4_sanity_check", f"Table E4: sanity check, rank correlation with the original map ({len(ids_full)} frames, model 1)",
              "Absolute values of the signed maps are compared. Layers are randomized from the output towards the input.\n"
              "A map that really depends on the model loses its correlation (towards 0). A map that keeps a high correlation after full randomization would only be showing the image.",
              pd.DataFrame(sanity_rows))
else:
    print("no detected frames, sanity check skipped")

# Table E5: checks in one place
checks = pd.DataFrame([
    {"check": "Head-CAM adds up to head output (largest error)", "result": f"{completeness_error:.1e}",
     "meaning": "below 1e-4 = the decomposition is exact"},
    {"check": "frames detected by model 1 among sampled nerve frames", "result": f"{detected.mean():.2f}",
     "meaning": "explanations are most meaningful on detected frames"},
    {"check": "Grad-CAM maps that are completely zero", "result": f"{scores_by_method['Grad-CAM']['empty'].mean():.2f}",
     "meaning": "no positive evidence found for these frames"}])
add_table("E5_checks", "Table E5: technical checks", "", checks)


##### 5. Plots
saved_plots = []
def save_fig(fig, name):
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    saved_plots.append(path)

# One fixed colour scale per method: the 99.5th percentile over the sampled nerve frames.
# Red = as strong as on a typical nerve frame. A frame with weak evidence stays blue
# (stretching every map to its own maximum would make noise on empty frames look important).
MAP_SCALE = {m: max(float(np.percentile(maps[m], 99.5)), 1e-12) for m in METHODS}

def overlay(ax, image, mp, title, vmax):
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    if mp is not None:
        ax.imshow(np.clip(mp / vmax, 0, 1), cmap="jet", alpha=0.45, vmin=0, vmax=1)
    ax.set_title(title, fontsize=7)
    ax.axis("off")

# X1: gallery. Frames are drawn at random (seeded) from the four outcome groups, not picked by eye.
p0 = presence_prob(model0, X[test_idx])
groups = {"nerve, detected": np.where(test_has & (p0 >= PRES_THR))[0], "nerve, missed": np.where(test_has & (p0 < PRES_THR))[0],
          "empty, false alarm": np.where(~test_has & (p0 >= PRES_THR))[0], "empty, correct": np.where(~test_has & (p0 < PRES_THR))[0]}
gallery = [(name, i) for name, ids in groups.items() for i in rng.permutation(ids)[:2]]
if gallery:
    gids = np.array([i for _, i in gallery])
    gmaps, gprob, _ = gradient_maps(model0, test_idx[gids])
    gmaps["Occlusion"] = np.stack([occlusion_map(model0, X[test_idx[i]]) for i in gids])
    with torch.no_grad():
        gseg = torch.sigmoid(model0(torch.from_numpy(X[test_idx[gids]]).unsqueeze(1).to(device))[0])[:, 0].cpu().numpy()
    fig, axes = plt.subplots(len(gallery), 6, figsize=(12, 2.0 * len(gallery)))
    axes = np.atleast_2d(axes)
    for r, (name, i) in enumerate(gallery):
        image = X[test_idx[i]]
        axes[r, 0].imshow(image, cmap="gray", vmin=0, vmax=1)
        if test_true[i].any():
            axes[r, 0].contour(test_true[i].astype(float), levels=[0.5], colors="#2ecc71", linewidths=1.2)
        axes[r, 0].set_title(f"{name}\nhead {gprob[r]:.2f}", fontsize=7); axes[r, 0].axis("off")
        overlay(axes[r, 1], image, gseg[r], "predicted nerve probability", 1.0)
        for c, method in enumerate(METHODS):
            overlay(axes[r, 2 + c], image, gmaps[method][r], method, MAP_SCALE[method])
    fig.suptitle("X1  What the model looks at. Green = true nerve. Colour = importance on one fixed scale per method "
                 "(red = as strong as on typical nerve frames).\nFrames drawn at random from each outcome group (test patients, model 1).", fontsize=9, y=1.04)
    save_fig(fig, "X1_gallery.png")

# X2: map quality with intervals
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
quality = [(m, scores_by_method[m]) for m in METHODS + ["Random"]]
for ax, key, title in [(axes[0], "hit", "pointing game: share of frames where the peak is on the nerve"),
                       (axes[1], "enrichment", "energy / nerve share (1 = evenly spread)")]:
    for j, (m, s) in enumerate(quality):
        mean, lo, hi = boot_mean(s[key], sub_pat)
        ax.errorbar(j, mean, yerr=[[mean - lo], [hi - mean]], fmt="o", color=METHOD_COLORS[m], capsize=3)
    ax.set_xticks(range(len(quality))); ax.set_xticklabels([m for m, _ in quality], rotation=20, fontsize=8)
    ax.set_title(title, fontsize=8)
axes[0].axhline(np.nanmean(scores_by_method["Random"]["chance"]), color=C_RED, ls="--", lw=0.8)
axes[0].text(len(quality) - 1.4, np.nanmean(scores_by_method["Random"]["chance"]) + 0.01, "chance", color=C_RED, fontsize=7)
axes[1].axhline(1, color=C_RED, ls="--", lw=0.8)
fig.suptitle("X2  Do the maps point at the true nerve? (all sampled nerve frames, 95% interval over patients)", fontsize=9, y=1.04)
save_fig(fig, "X2_map_quality.png")

# X3: deletion curves
if deletion:
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    for method in METHODS + ["Random"]:
        c = deletion[method]
        ax.errorbar(np.array(FRACTIONS) * 100, c.mean(axis=0), yerr=c.std(axis=0) / np.sqrt(len(c)), color=METHOD_COLORS[method],
                    lw=1.5 if method != "Random" else 1.2, ls="-" if method != "Random" else "--", marker="o", ms=3, label=method)
    ax.set_xlabel("share of pixels removed, most important first (%)"); ax.set_ylabel("detection probability")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("X3  Deletion test. Falling faster than the dashed random curve = the map is faithful.", fontsize=8)
    save_fig(fig, "X3_deletion.png")

# X4: weight-randomization check
if len(sanity_local) > 0:
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for m, vals in sanity_curves.items():
        mean = np.array([v[0] for v in vals]); sd = np.array([v[1] for v in vals])
        ax.errorbar(range(len(mean)), mean, yerr=sd, marker="o", ms=3, capsize=2, color=METHOD_COLORS[m], label=m)
    ax.axhline(0, color=C_GREY, lw=0.8)
    ax.set_xticks(range(len(levels))); ax.set_xticklabels(levels, rotation=25, fontsize=7)
    ax.set_ylabel("rank correlation with original map"); ax.legend(frameon=False, fontsize=7)
    ax.set_title("X4  Sanity check. A map that depends on the model falls towards 0 when the weights are randomized.", fontsize=8)
    save_fig(fig, "X4_sanity_check.png")

# X5: where in the image (depth) does the model look? Row 0 is near the probe.
fig, ax = plt.subplots(figsize=(5.8, 3.8))
rows_axis = np.arange(H)
truth_profile = sub_true.sum(axis=2).sum(axis=0).astype(float)
ax.plot(truth_profile / truth_profile.sum(), rows_axis, color="#2ecc71", lw=2, label="true nerve")
for method in METHODS:
    prof = maps[method].sum(axis=2) / np.maximum(maps[method].sum(axis=(1, 2))[:, None], 1e-12)
    ax.plot(prof.mean(axis=0), rows_axis, color=METHOD_COLORS[method], label=method)
ax.invert_yaxis(); ax.set_ylabel("image row (0 = near the probe)"); ax.set_xlabel("share of map / nerve in this row")
ax.legend(frameon=False, fontsize=7)
ax.set_title("X5  Depth profile. Maps far from the green curve rely on something other than the nerve.", fontsize=8)
save_fig(fig, "X5_depth_profile.png")


##### 6. Report and MLflow

with open(os.path.join(TABLE_DIR, "explain_report.md"), "w") as f:
    f.write("# Explainability report (step 10)\n\n")
    f.write("Model 1 unless stated otherwise. Test patients only. Maps explain one forward pass without flip averaging.\n\n")
    for title, note, df in report_parts:
        f.write(f"## {title}\n\n")
        if note:
            f.write(note.replace("\n", "  \n") + "\n\n")
        f.write(md_table(df) + "\n\n")

for method in METHODS:
    s = scores_by_method[method]
    mlflow.log_metric(f"pointing_hit_{method.lower().replace('-', '_')}", float(np.nanmean(s["hit"])))
    mlflow.log_metric(f"energy_ratio_{method.lower().replace('-', '_')}", float(np.nanmean(s["enrichment"])))
mlflow.log_metric("headcam_completeness_error", completeness_error)
mlflow.log_artifacts(TABLE_DIR, artifact_path="tables")
mlflow.log_artifacts(PLOT_DIR, artifact_path="plots")
try:
    mlflow.log_artifact(__file__)
except NameError:
    pass
mlflow.end_run()
print(f"\nsaved {len(saved_plots)} plots to {PLOT_DIR}; the tables are in {TABLE_DIR} (explain_report.md has all tables)")##### Step 10: explainability (what does the model look at?)
# Reads the models, the split and final_config.json from steps 8 and 9. Trains nothing.
#
# Four ways to look at the model, all for the question "why does it say nerve / where is the nerve?":
#   Grad-CAM        gradient of the detection-head output with respect to the deepest layer (quarter size)
#   Head-CAM        the same gradient multiplied with the layer, without averaging. The head is linear
#                   (mean + max of the layer -> 1 number), so this splits the head output EXACTLY into
#                   one contribution per location. Check: the contributions add up to the output.
#   Seg-Grad-CAM    like Grad-CAM, but for the segmentation: gradient of the summed mask logits
#                   (inside the predicted mask) with respect to the last decoder layer (full size).
#                   The segmentation also uses skip connections that bypass the deepest layer,
#                   so the deepest layer alone would not explain it.
#   Occlusion       model-agnostic: hide a 16x16 patch, measure how much the head output drops
# A picture of a map proves nothing. So the maps are tested with numbers:
#   pointing game   does the strongest point of the map lie on the true nerve (3 pixel tolerance)?
#   energy in nerve how much of the map lies on the true nerve, divided by the nerve's share of the image
#                   (1 = no better than spreading the map evenly)
#   deletion        remove the most important pixels first: does the detection drop faster than when
#                   random pixels are removed? (faithfulness)
#   sanity check    randomize the weights layer by layer: the maps must change. If they do not, the
#                   map describes the image and not the model (Adebayo et al., 2018).
#   edge reliance   how much of the map lies in the outer 10% of the image (probe contact, borders)?
# Explanations are for one forward pass of one fold model, without flip averaging.
# Maps are shown on the test patients, which were not used for training or for choosing anything.
#
# Output (in OUT_DIR): tables/E*.csv, tables/explain_report.md, plots/X*.png
# Needs:  pip install mlflow torch scikit-learn scipy pandas matplotlib

import json
import os
import time
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)       # all-NaN slices in the interval calculation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from scipy.stats import spearmanr

OUT_DIR = os.environ.get("OUT_DIR", "outputs")
H, W = 96, 128
SEED = 0
N_EXPLAIN = int(os.environ.get("N_EXPLAIN", 150))     # test nerve frames used for occlusion, deletion, the main table
N_SANITY = int(os.environ.get("N_SANITY", 40))        # frames for the weight-randomization check
N_BOOT = int(os.environ.get("N_BOOT", 2000))
OCC, STRIDE = 16, 8                                   # occlusion patch and step, in pixels
TOLERANCE = 3                                         # pointing game: pixels around the true nerve that still count
BORDER = 0.10                                         # outer share of the image counted as "edge"
FRACTIONS = [0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]  # share of pixels removed in the deletion test
RUN_NAME = "step10_explain"

TABLE_DIR = os.path.join(OUT_DIR, "tables")
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
rng = np.random.default_rng(SEED)

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 130, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
C_NERVE, C_EMPTY, C_GREY, C_RED = "#1b6ca8", "#d9822b", "#8a8a8a", "#c0392b"
METHODS = ["Grad-CAM", "Head-CAM", "Seg-Grad-CAM", "Occlusion"]
METHOD_COLORS = {"Grad-CAM": "#1b6ca8", "Head-CAM": "#2a9d8f", "Seg-Grad-CAM": "#8e44ad", "Occlusion": "#d9822b", "Random": C_GREY}

mlflow.set_tracking_uri(f"sqlite:///{os.path.join(OUT_DIR, 'mlflow.db')}")
mlflow.set_experiment("nerve_segmentation")
mlflow.start_run(run_name=RUN_NAME)
mlflow.log_params({"step": 10, "n_explain": N_EXPLAIN, "n_sanity": N_SANITY, "occlusion_patch": OCC,
                   "occlusion_stride": STRIDE, "pointing_tolerance_px": TOLERANCE})


##### 1. Load the cache, the split, the final configuration and the models

cache = np.load(os.path.join(OUT_DIR, "data_cache.npz"))
X = cache["X"].astype(np.float32) / 255.0
Y = cache["Y"].astype(np.float32)
patients = cache["patients"]
has_nerve = Y.sum(axis=(1, 2)) > 0
split = np.load(os.path.join(OUT_DIR, "split.npz"))
test_idx, fold_of_dev = split["test_idx"], split["fold_of_dev"]
N_FOLDS = int(fold_of_dev.max()) + 1
test_true, test_has, test_pat = Y[test_idx] > 0.5, has_nerve[test_idx], patients[test_idx]
with open(os.path.join(OUT_DIR, "final_config.json")) as f:
    final_config = json.load(f)
PRES_THR = final_config["settings"]["pres_thr"]
print(f"test: {len(test_idx)} frames, {len(set(test_pat))} patients; detection threshold {PRES_THR}")

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
model0 = models[0]                      # the model used for occlusion, deletion, sanity check and the gallery


##### 2. The four explanation methods

def gradient_maps(model, ids, signed=False):
    """Grad-CAM, Head-CAM and Seg-Grad-CAM for the frames X[ids] (ids index into the full arrays).
    Returns maps of shape (n, H, W), plus the detection probability and the Head-CAM check.
    signed=False: negative values are cut to 0 (only evidence FOR a nerve), the normal setting.
    signed=True: the absolute value of the signed map is returned, used in the weight-randomization check."""
    out = {"Grad-CAM": [], "Head-CAM": [], "Seg-Grad-CAM": []}
    probs, completeness = [], []
    for start in range(0, len(ids), 64):
        xb = torch.from_numpy(X[ids[start:start + 64]]).unsqueeze(1).to(device)
        kept = {}
        hook = model.dec1.register_forward_hook(lambda module, inputs, output: kept.update(d1=output))
        seg_logits, pres_logit, m = model(xb, return_features=True)      # m: deepest layer, (B, 64, H/4, W/4)
        hook.remove()
        d1 = kept["d1"]                                                  # last decoder layer, (B, 16, H, W)

        # detection head: Grad-CAM (gradient averaged over space) and Head-CAM (gradient times layer, kept per location)
        g = torch.autograd.grad(pres_logit.sum(), m, retain_graph=True)[0]
        grad_cam = (g.mean(dim=(2, 3), keepdim=True) * m).sum(dim=1, keepdim=True)
        head_cam = (g * m).sum(dim=1, keepdim=True)
        # check: the contributions must add up to the head output minus its bias
        completeness.append((head_cam.sum(dim=(1, 2, 3)) - (pres_logit - model.presence.bias)).abs().detach().cpu().numpy())

        # segmentation: sum of mask logits inside the predicted mask (if empty: inside the model's best guess)
        prob = torch.sigmoid(seg_logits)
        region = prob > 0.5
        nothing = ~region.flatten(1).any(dim=1)
        best_guess = prob >= 0.8 * prob.amax(dim=(1, 2, 3), keepdim=True)
        region = torch.where(nothing[:, None, None, None], best_guess, region).float()
        g2 = torch.autograd.grad((seg_logits * region).sum(), d1)[0]
        seg_cam = (g2.mean(dim=(2, 3), keepdim=True) * d1).sum(dim=1, keepdim=True)

        for name, cam in [("Grad-CAM", grad_cam), ("Head-CAM", head_cam), ("Seg-Grad-CAM", seg_cam)]:
            if signed:
                up = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)[:, 0].abs()
            else:
                up = F.interpolate(F.relu(cam), size=(H, W), mode="bilinear", align_corners=False)[:, 0]
            out[name].append(up.detach().cpu().numpy())
        probs.append(torch.sigmoid(pres_logit).detach().cpu().numpy())
    return ({k: np.concatenate(v) for k, v in out.items()}, np.concatenate(probs), float(np.concatenate(completeness).max()))

def occlusion_map(model, image):
    """Drop of the detection logit when a OCC x OCC patch is replaced by the frame mean. image: (H, W)."""
    positions = [(y, x) for y in range(0, H - OCC + 1, STRIDE) for x in range(0, W - OCC + 1, STRIDE)]
    batch = np.repeat(image[None, None], len(positions) + 1, axis=0)          # last entry stays unchanged
    for i, (y, x) in enumerate(positions):
        batch[i, 0, y:y + OCC, x:x + OCC] = image.mean()
    with torch.no_grad():
        logits = model(torch.from_numpy(batch).to(device))[1].cpu().numpy()
    drop = logits[-1] - logits[:-1]
    total, count = np.zeros((H, W)), np.zeros((H, W))
    for (y, x), d in zip(positions, drop):
        total[y:y + OCC, x:x + OCC] += d
        count[y:y + OCC, x:x + OCC] += 1
    return np.maximum(total / np.maximum(count, 1), 0).astype(np.float32)   # only "this region supports a nerve"

def presence_prob(model, images):
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(images).unsqueeze(1).to(device))[1]).cpu().numpy()


##### 3. Maps for the test nerve frames

nerve_pos = np.where(test_has)[0]                                           # positions inside the test arrays
subset = np.sort(rng.choice(nerve_pos, size=min(N_EXPLAIN, len(nerve_pos)), replace=False))
t0 = time.time()
maps, probs0, completeness_error = gradient_maps(model0, test_idx[subset])
maps["Occlusion"] = np.stack([occlusion_map(model0, X[test_idx[i]]) for i in subset])
maps["Random"] = rng.random((len(subset), H, W)).astype(np.float32)        # baseline: a map with no information
print(f"maps for {len(subset)} test nerve frames took {time.time() - t0:.0f} s")
print(f"Head-CAM check: largest |sum of contributions - (output - bias)| = {completeness_error:.2e}")
detected = probs0 >= PRES_THR                                               # frames where the model says "nerve"
sub_true, sub_pat = test_true[subset], test_pat[subset]
print(f"{detected.sum()} of {len(subset)} frames are detected by model 1 (threshold {PRES_THR})")


##### 4. Numbers for the maps

border_mask = np.zeros((H, W), dtype=bool)
bh, bw = int(round(BORDER * H)), int(round(BORDER * W))
border_mask[:bh], border_mask[-bh:], border_mask[:, :bw], border_mask[:, -bw:] = True, True, True, True

def map_scores(map_array, true):
    """Per-frame scores of a map (n, H, W) against the true nerve masks."""
    n = len(map_array)
    hit, energy, enrich, chance, edge, empty = (np.full(n, np.nan) for _ in range(6))
    for i in range(n):
        mp = map_array[i]
        grown = ndimage.binary_dilation(true[i], iterations=TOLERANCE)
        chance[i] = grown.mean()
        total = mp.sum()
        empty[i] = float(mp.max() <= 0)
        hit[i] = float(mp.max() > 0 and grown[np.unravel_index(np.argmax(mp), mp.shape)])
        if total > 0:
            energy[i] = mp[true[i]].sum() / total
            enrich[i] = energy[i] / true[i].mean()
            edge[i] = (mp[border_mask].sum() / total) / border_mask.mean()
    return {"hit": hit, "energy": energy, "enrichment": enrich, "chance": chance, "edge": edge, "empty": empty}

def boot_mean(values, groups, n_boot=N_BOOT):
    """Mean with a 95% interval; whole patients are redrawn."""
    draw_rng = np.random.default_rng(SEED)
    ids = np.unique(groups)
    members = [np.where(groups == g)[0] for g in ids]
    boots = []
    for _ in range(n_boot):
        drawn = draw_rng.integers(0, len(ids), len(ids))
        idx = np.concatenate([members[j] for j in drawn])
        boots.append(np.nanmean(values[idx]) if np.isfinite(values[idx]).any() else np.nan)
    return float(np.nanmean(values)), float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))

def fmt(m, lo, hi, digits=2):
    return f"{m:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"

def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)

report_parts = []
def add_table(name, title, note, df):
    df.to_csv(os.path.join(TABLE_DIR, name + ".csv"), index=False)
    report_parts.append((title, note, df))
    print(f"\n=== {title} ===")
    if note:
        print(note)
    print(df.to_string(index=False))

# Table E1: pointing game, energy in nerve, edge reliance (model 1, the sampled test nerve frames)
e1_rows, scores_by_method = [], {}
for method in METHODS + ["Random"]:
    scores_by_method[method] = map_scores(maps[method], sub_true)
for subset_name, keep in [("all sampled nerve frames", np.ones(len(subset), dtype=bool)), ("frames the model detects", detected)]:
    for method in METHODS + ["Random"]:
        s = scores_by_method[method]
        if keep.sum() == 0:
            continue
        row = {"method": method, "frames": subset_name, "n": int(keep.sum()),
               "pointing hit": fmt(*boot_mean(s["hit"][keep], sub_pat[keep])),
               "chance": f"{np.nanmean(s['chance'][keep]):.2f}",
               "energy in nerve": fmt(*boot_mean(s["energy"][keep], sub_pat[keep])),
               "energy / nerve share": fmt(*boot_mean(s["enrichment"][keep], sub_pat[keep]), digits=1),
               "edge reliance": f"{np.nanmean(s['edge'][keep]):.2f}",
               "empty maps": f"{np.mean(s['empty'][keep]):.2f}"}
        e1_rows.append(row)
add_table("E1_map_quality", "Table E1: do the maps point at the nerve? (model 1, test patients, value [95% interval over patients])",
          "pointing hit: strongest point lies within 3 pixels of the true nerve. chance: hit rate of a random point.\n"
          "energy / nerve share: 1 = map is spread evenly, higher = map is concentrated on the nerve.\n"
          "edge reliance: map share in the outer 10% of the image divided by the area share (1 = neutral, above 1 = relies on the edges).\n"
          "Random = a map with no information, shown as a reference.", pd.DataFrame(e1_rows))

# Table E2: the same for the 3 gradient methods on ALL test nerve frames, one row per model (stability across models)
e2_rows = []
for k, model in enumerate(models):
    mk, pk, _ = gradient_maps(model, test_idx[nerve_pos])
    for method in ["Grad-CAM", "Head-CAM", "Seg-Grad-CAM"]:
        s = map_scores(mk[method], test_true[nerve_pos])
        e2_rows.append({"model": k + 1, "method": method, "frames": len(nerve_pos), "detected": f"{(pk >= PRES_THR).mean():.2f}",
                        "pointing hit": f"{np.nanmean(s['hit']):.3f}", "energy / nerve share": f"{np.nanmean(s['enrichment']):.1f}"})
e2_df = pd.DataFrame(e2_rows)
spread = e2_df.assign(hit=e2_df["pointing hit"].astype(float)).groupby("method").hit.agg(["mean", "std"]).reset_index()
add_table("E2_model_stability", "Table E2: the same test per fold model (all test nerve frames)",
          "If the pointing hit differs a lot between models, the explanation depends on which patients the model was trained on.", e2_df)
print("pointing hit, mean +- sd over the models:")
print(spread.to_string(index=False))

# Table E3: deletion test (faithfulness)
def delete_curve(ranking, ids_local):
    """Detection probability of model 1 after removing the top-ranked pixels (replaced by the frame mean)."""
    curves = np.zeros((len(ids_local), len(FRACTIONS)))
    for j, i in enumerate(ids_local):
        image = X[test_idx[subset[i]]]
        order = np.argsort(-(ranking[i].ravel() + 1e-9 * rng.random(H * W)))        # tiny noise breaks ties
        batch = np.repeat(image[None], len(FRACTIONS), axis=0)
        for f_i, frac in enumerate(FRACTIONS):
            kill = order[:int(round(frac * H * W))]
            flat = batch[f_i].reshape(-1)
            flat[kill] = image.mean()
        curves[j] = presence_prob(model0, batch)
    return curves

del_ids = np.where(detected)[0]
deletion = {}
if len(del_ids) > 0:
    for method in METHODS:
        deletion[method] = delete_curve(maps[method], del_ids)
    deletion["Random"] = np.mean([delete_curve(rng.random((len(subset), H, W)), del_ids) for _ in range(3)], axis=0)
    area = {m: np.trapezoid(c, FRACTIONS, axis=1) / FRACTIONS[-1] for m, c in deletion.items()}   # mean detection prob. over the curve
    del_rows = []
    for method in METHODS + ["Random"]:
        row = {"method": method, "n": len(del_ids), "detection after removing 5%": f"{deletion[method][:, 2].mean():.2f}",
               "after 20%": f"{deletion[method][:, 4].mean():.2f}", "mean over the curve (lower = better)": fmt(*boot_mean(area[method], sub_pat[del_ids]), digits=3)}
        if method != "Random":
            row["difference to random"] = fmt(*boot_mean(area[method] - area["Random"], sub_pat[del_ids]), digits=3)
        else:
            row["difference to random"] = "-"
        del_rows.append(row)
    add_table("E3_deletion", "Table E3: deletion test (model 1, frames it detects)",
              "The most important pixels are removed first. A faithful map makes the detection probability fall faster than random removal.\n"
              "A difference to random below 0 with an interval that stays below 0 = better than random.", pd.DataFrame(del_rows))
else:
    print("no detected frames, deletion test skipped")

# Table E4: sanity check, randomize the weights from the output back to the input
def randomized_copy(model, upto):
    """Copy of the model with fresh random weights in the first `upto` stages, counted from the output."""
    stages = [("heads", [model.out, model.presence]), ("dec1", [model.up1, model.dec1]), ("dec2", [model.up2, model.dec2]),
              ("middle", [model.middle]), ("enc2", [model.enc2]), ("enc1", [model.enc1])]
    clone = UNet().to(device)
    clone.load_state_dict(model.state_dict())
    torch.manual_seed(SEED + 100)
    parts = {"heads": [clone.out, clone.presence], "dec1": [clone.up1, clone.dec1], "dec2": [clone.up2, clone.dec2],
             "middle": [clone.middle], "enc2": [clone.enc2], "enc1": [clone.enc1]}
    for name, _ in stages[:upto]:
        for part in parts[name]:
            for sub in part.modules():
                if hasattr(sub, "reset_parameters"):
                    sub.reset_parameters()
    return clone.eval()

sanity_local = np.where(detected)[0][:N_SANITY]
levels = ["original", "heads", "+ dec1", "+ dec2", "+ middle", "+ enc2", "+ enc1 (all)"]
sanity_rows, sanity_curves = [], {m: [] for m in ["Grad-CAM", "Head-CAM", "Seg-Grad-CAM"]}
if len(sanity_local) > 0:
    ids_full = test_idx[subset[sanity_local]]
    reference, _, _ = gradient_maps(model0, ids_full, signed=True)
    for level in range(0, 7):
        if level == 0:
            row = {"randomized layers": levels[0]}
            for m in sanity_curves:
                row[m] = "1.00"
                sanity_curves[m].append((1.0, 0.0))
        else:
            clone = randomized_copy(model0, level)
            rand_maps, _, _ = gradient_maps(clone, ids_full, signed=True)
            row = {"randomized layers": levels[level]}
            for m in sanity_curves:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    rho = np.array([spearmanr(reference[m][i].ravel(), rand_maps[m][i].ravel()).correlation for i in range(len(ids_full))])
                ok = np.isfinite(rho)
                mean_rho, sd_rho = (float(rho[ok].mean()), float(rho[ok].std())) if ok.any() else (np.nan, np.nan)
                row[m] = f"{mean_rho:.2f} +- {sd_rho:.2f}" if ok.any() else "map is empty"
                sanity_curves[m].append((mean_rho, sd_rho))
        sanity_rows.append(row)
    add_table("E4_sanity_check", f"Table E4: sanity check, rank correlation with the original map ({len(ids_full)} frames, model 1)",
              "Absolute values of the signed maps are compared. Layers are randomized from the output towards the input.\n"
              "A map that really depends on the model loses its correlation (towards 0). A map that keeps a high correlation after full randomization would only be showing the image.",
              pd.DataFrame(sanity_rows))
else:
    print("no detected frames, sanity check skipped")

# Table E5: checks in one place
checks = pd.DataFrame([
    {"check": "Head-CAM adds up to head output (largest error)", "result": f"{completeness_error:.1e}",
     "meaning": "below 1e-4 = the decomposition is exact"},
    {"check": "frames detected by model 1 among sampled nerve frames", "result": f"{detected.mean():.2f}",
     "meaning": "explanations are most meaningful on detected frames"},
    {"check": "Grad-CAM maps that are completely zero", "result": f"{scores_by_method['Grad-CAM']['empty'].mean():.2f}",
     "meaning": "no positive evidence found for these frames"}])
add_table("E5_checks", "Table E5: technical checks", "", checks)


##### 5. Plots
saved_plots = []
def save_fig(fig, name):
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    saved_plots.append(path)

# One fixed colour scale per method: the 99.5th percentile over the sampled nerve frames.
# Red = as strong as on a typical nerve frame. A frame with weak evidence stays blue
# (stretching every map to its own maximum would make noise on empty frames look important).
MAP_SCALE = {m: max(float(np.percentile(maps[m], 99.5)), 1e-12) for m in METHODS}

def overlay(ax, image, mp, title, vmax):
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    if mp is not None:
        ax.imshow(np.clip(mp / vmax, 0, 1), cmap="jet", alpha=0.45, vmin=0, vmax=1)
    ax.set_title(title, fontsize=7)
    ax.axis("off")

# X1: gallery. Frames are drawn at random (seeded) from the four outcome groups, not picked by eye.
p0 = presence_prob(model0, X[test_idx])
groups = {"nerve, detected": np.where(test_has & (p0 >= PRES_THR))[0], "nerve, missed": np.where(test_has & (p0 < PRES_THR))[0],
          "empty, false alarm": np.where(~test_has & (p0 >= PRES_THR))[0], "empty, correct": np.where(~test_has & (p0 < PRES_THR))[0]}
gallery = [(name, i) for name, ids in groups.items() for i in rng.permutation(ids)[:2]]
if gallery:
    gids = np.array([i for _, i in gallery])
    gmaps, gprob, _ = gradient_maps(model0, test_idx[gids])
    gmaps["Occlusion"] = np.stack([occlusion_map(model0, X[test_idx[i]]) for i in gids])
    with torch.no_grad():
        gseg = torch.sigmoid(model0(torch.from_numpy(X[test_idx[gids]]).unsqueeze(1).to(device))[0])[:, 0].cpu().numpy()
    fig, axes = plt.subplots(len(gallery), 6, figsize=(12, 2.0 * len(gallery)))
    axes = np.atleast_2d(axes)
    for r, (name, i) in enumerate(gallery):
        image = X[test_idx[i]]
        axes[r, 0].imshow(image, cmap="gray", vmin=0, vmax=1)
        if test_true[i].any():
            axes[r, 0].contour(test_true[i].astype(float), levels=[0.5], colors="#2ecc71", linewidths=1.2)
        axes[r, 0].set_title(f"{name}\nhead {gprob[r]:.2f}", fontsize=7); axes[r, 0].axis("off")
        overlay(axes[r, 1], image, gseg[r], "predicted nerve probability", 1.0)
        for c, method in enumerate(METHODS):
            overlay(axes[r, 2 + c], image, gmaps[method][r], method, MAP_SCALE[method])
    fig.suptitle("X1  What the model looks at. Green = true nerve. Colour = importance on one fixed scale per method "
                 "(red = as strong as on typical nerve frames).\nFrames drawn at random from each outcome group (test patients, model 1).", fontsize=9, y=1.04)
    save_fig(fig, "X1_gallery.png")

# X2: map quality with intervals
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
quality = [(m, scores_by_method[m]) for m in METHODS + ["Random"]]
for ax, key, title in [(axes[0], "hit", "pointing game: share of frames where the peak is on the nerve"),
                       (axes[1], "enrichment", "energy / nerve share (1 = evenly spread)")]:
    for j, (m, s) in enumerate(quality):
        mean, lo, hi = boot_mean(s[key], sub_pat)
        ax.errorbar(j, mean, yerr=[[mean - lo], [hi - mean]], fmt="o", color=METHOD_COLORS[m], capsize=3)
    ax.set_xticks(range(len(quality))); ax.set_xticklabels([m for m, _ in quality], rotation=20, fontsize=8)
    ax.set_title(title, fontsize=8)
axes[0].axhline(np.nanmean(scores_by_method["Random"]["chance"]), color=C_RED, ls="--", lw=0.8)
axes[0].text(len(quality) - 1.4, np.nanmean(scores_by_method["Random"]["chance"]) + 0.01, "chance", color=C_RED, fontsize=7)
axes[1].axhline(1, color=C_RED, ls="--", lw=0.8)
fig.suptitle("X2  Do the maps point at the true nerve? (all sampled nerve frames, 95% interval over patients)", fontsize=9, y=1.04)
save_fig(fig, "X2_map_quality.png")

# X3: deletion curves
if deletion:
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    for method in METHODS + ["Random"]:
        c = deletion[method]
        ax.errorbar(np.array(FRACTIONS) * 100, c.mean(axis=0), yerr=c.std(axis=0) / np.sqrt(len(c)), color=METHOD_COLORS[method],
                    lw=1.5 if method != "Random" else 1.2, ls="-" if method != "Random" else "--", marker="o", ms=3, label=method)
    ax.set_xlabel("share of pixels removed, most important first (%)"); ax.set_ylabel("detection probability")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("X3  Deletion test. Falling faster than the dashed random curve = the map is faithful.", fontsize=8)
    save_fig(fig, "X3_deletion.png")

# X4: weight-randomization check
if len(sanity_local) > 0:
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for m, vals in sanity_curves.items():
        mean = np.array([v[0] for v in vals]); sd = np.array([v[1] for v in vals])
        ax.errorbar(range(len(mean)), mean, yerr=sd, marker="o", ms=3, capsize=2, color=METHOD_COLORS[m], label=m)
    ax.axhline(0, color=C_GREY, lw=0.8)
    ax.set_xticks(range(len(levels))); ax.set_xticklabels(levels, rotation=25, fontsize=7)
    ax.set_ylabel("rank correlation with original map"); ax.legend(frameon=False, fontsize=7)
    ax.set_title("X4  Sanity check. A map that depends on the model falls towards 0 when the weights are randomized.", fontsize=8)
    save_fig(fig, "X4_sanity_check.png")

# X5: where in the image (depth) does the model look? Row 0 is near the probe.
fig, ax = plt.subplots(figsize=(5.8, 3.8))
rows_axis = np.arange(H)
truth_profile = sub_true.sum(axis=2).sum(axis=0).astype(float)
ax.plot(truth_profile / truth_profile.sum(), rows_axis, color="#2ecc71", lw=2, label="true nerve")
for method in METHODS:
    prof = maps[method].sum(axis=2) / np.maximum(maps[method].sum(axis=(1, 2))[:, None], 1e-12)
    ax.plot(prof.mean(axis=0), rows_axis, color=METHOD_COLORS[method], label=method)
ax.invert_yaxis(); ax.set_ylabel("image row (0 = near the probe)"); ax.set_xlabel("share of map / nerve in this row")
ax.legend(frameon=False, fontsize=7)
ax.set_title("X5  Depth profile. Maps far from the green curve rely on something other than the nerve.", fontsize=8)
save_fig(fig, "X5_depth_profile.png")


##### 6. Report and MLflow

with open(os.path.join(TABLE_DIR, "explain_report.md"), "w") as f:
    f.write("# Explainability report (step 10)\n\n")
    f.write("Model 1 unless stated otherwise. Test patients only. Maps explain one forward pass without flip averaging.\n\n")
    for title, note, df in report_parts:
        f.write(f"## {title}\n\n")
        if note:
            f.write(note.replace("\n", "  \n") + "\n\n")
        f.write(md_table(df) + "\n\n")

for method in METHODS:
    s = scores_by_method[method]
    mlflow.log_metric(f"pointing_hit_{method.lower().replace('-', '_')}", float(np.nanmean(s["hit"])))
    mlflow.log_metric(f"energy_ratio_{method.lower().replace('-', '_')}", float(np.nanmean(s["enrichment"])))
mlflow.log_metric("headcam_completeness_error", completeness_error)
mlflow.log_artifacts(TABLE_DIR, artifact_path="tables")
mlflow.log_artifacts(PLOT_DIR, artifact_path="plots")
try:
    mlflow.log_artifact(__file__)
except NameError:
    pass
mlflow.end_run()
print(f"\nsaved {len(saved_plots)} plots to {PLOT_DIR}; the tables are in {TABLE_DIR} (explain_report.md has all tables)")

############ OUTPUT ############

'''
test: 1199 frames, 10 patients; detection threshold 0.6
maps for 150 test nerve frames took 14 s
Head-CAM check: largest |sum of contributions - (output - bias)| = 8.94e-07
57 of 150 frames are detected by model 1 (threshold 0.6)

=== Table E1: do the maps point at the nerve? (model 1, test patients, value [95% interval over patients]) ===
pointing hit: strongest point lies within 3 pixels of the true nerve. chance: hit rate of a random point.
energy / nerve share: 1 = map is spread evenly, higher = map is concentrated on the nerve.
edge reliance: map share in the outer 10% of the image divided by the area share (1 = neutral, above 1 = relies on the edges).
Random = a map with no information, shown as a reference.
      method                   frames   n      pointing hit chance   energy in nerve energy / nerve share edge reliance empty maps
    Grad-CAM all sampled nerve frames 150 0.79 [0.64, 0.89]   0.05 0.73 [0.61, 0.81]    26.1 [21.0, 30.8]          0.01       0.05
    Head-CAM all sampled nerve frames 150 0.00 [0.00, 0.00]   0.05 0.02 [0.01, 0.02]       0.7 [0.5, 0.9]          2.62       0.00
Seg-Grad-CAM all sampled nerve frames 150 0.82 [0.68, 0.91]   0.05 0.60 [0.51, 0.66]    21.1 [17.6, 24.7]          0.00       0.01
   Occlusion all sampled nerve frames 150 0.01 [0.00, 0.02]   0.05 0.04 [0.03, 0.05]       1.5 [1.1, 1.9]          1.65       0.00
      Random all sampled nerve frames 150 0.05 [0.02, 0.08]   0.05 0.03 [0.03, 0.03]       1.0 [1.0, 1.0]          1.00       0.00
    Grad-CAM frames the model detects  57 0.95 [0.87, 1.00]   0.05 0.83 [0.77, 0.89]    29.1 [24.6, 36.5]          0.01       0.00
    Head-CAM frames the model detects  57 0.00 [0.00, 0.00]   0.05 0.02 [0.02, 0.03]       0.8 [0.6, 1.0]          2.62       0.00
Seg-Grad-CAM frames the model detects  57 0.91 [0.79, 0.96]   0.05 0.65 [0.57, 0.69]    22.7 [17.8, 28.5]          0.00       0.00
   Occlusion frames the model detects  57 0.00 [0.00, 0.00]   0.05 0.03 [0.02, 0.04]       1.1 [0.8, 1.2]          1.90       0.00
      Random frames the model detects  57 0.04 [0.00, 0.10]   0.05 0.03 [0.02, 0.03]       1.0 [1.0, 1.0]          1.00       0.00

=== Table E2: the same test per fold model (all test nerve frames) ===
If the pointing hit differs a lot between models, the explanation depends on which patients the model was trained on.
 model       method  frames detected pointing hit energy / nerve share
     1     Grad-CAM     607     0.37        0.758                 26.6
     1     Head-CAM     607     0.37        0.000                  0.8
     1 Seg-Grad-CAM     607     0.37        0.830                 21.2
     2     Grad-CAM     607     0.25        0.680                 14.0
     2     Head-CAM     607     0.25        0.580                 10.7
     2 Seg-Grad-CAM     607     0.25        0.634                 28.8
     3     Grad-CAM     607     0.34        0.030                  1.5
     3     Head-CAM     607     0.34        0.437                  6.1
     3 Seg-Grad-CAM     607     0.34        0.840                 27.6
     4     Grad-CAM     607     0.29        0.807                 28.7
     4     Head-CAM     607     0.29        0.351                  9.9
     4 Seg-Grad-CAM     607     0.29        0.539                 31.5
     5     Grad-CAM     607     0.36        0.764                 19.7
     5     Head-CAM     607     0.36        0.713                 15.1
     5 Seg-Grad-CAM     607     0.36        0.646                 28.3
pointing hit, mean +- sd over the models:
      method   mean      std
    Grad-CAM 0.6078 0.326230
    Head-CAM 0.4162 0.270567
Seg-Grad-CAM 0.6978 0.131974

=== Table E3: deletion test (model 1, frames it detects) ===
The most important pixels are removed first. A faithful map makes the detection probability fall faster than random removal.
A difference to random below 0 with an interval that stays below 0 = better than random.
      method  n detection after removing 5% after 20% mean over the curve (lower = better)    difference to random
    Grad-CAM 57                        0.68      0.64                 0.630 [0.604, 0.648]  -0.005 [-0.011, 0.007]
    Head-CAM 57                        0.56      0.58                 0.585 [0.545, 0.598] -0.049 [-0.067, -0.040]
Seg-Grad-CAM 57                        0.69      0.64                 0.629 [0.604, 0.639] -0.006 [-0.011, -0.001]
   Occlusion 57                        0.49      0.42                 0.426 [0.396, 0.470] -0.209 [-0.228, -0.165]
      Random 57                        0.69      0.64                 0.635 [0.612, 0.645]                       -

=== Table E4: sanity check, rank correlation with the original map (40 frames, model 1) ===
Absolute values of the signed maps are compared. Layers are randomized from the output towards the input.
A map that really depends on the model loses its correlation (towards 0). A map that keeps a high correlation after full randomization would only be showing the image.
randomized layers      Grad-CAM      Head-CAM Seg-Grad-CAM
         original          1.00          1.00         1.00
            heads  0.61 +- 0.06  0.95 +- 0.02 1.00 +- 0.00
           + dec1  0.61 +- 0.06  0.95 +- 0.02 0.45 +- 0.03
           + dec2  0.61 +- 0.06  0.95 +- 0.02 0.02 +- 0.03
         + middle  0.36 +- 0.05  0.38 +- 0.06 0.03 +- 0.03
           + enc2 -0.03 +- 0.07 -0.28 +- 0.06 0.03 +- 0.03
     + enc1 (all)  0.28 +- 0.11 -0.26 +- 0.09 0.04 +- 0.03

=== Table E5: technical checks ===
                                                check  result                                             meaning
      Head-CAM adds up to head output (largest error) 8.9e-07             below 1e-4 = the decomposition is exact
frames detected by model 1 among sampled nerve frames    0.38 explanations are most meaningful on detected frames
               Grad-CAM maps that are completely zero    0.05         no positive evidence found for these frames

saved 5 plots to outputs/plots; the tables are in outputs/tables (explain_report.md has all tables)


'''
