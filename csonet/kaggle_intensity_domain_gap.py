# =============================================================================
# CSoNet track - Kaggle notebook: INTENSITY domain-gap on raw SPIDER images
# -----------------------------------------------------------------------------
# Purpose: complement the local ANATOMY domain-gap result. Download raw SPIDER
# MR images (Zenodo 10159290), extract intensity features inside the segmented
# structures, and measure how predictable the scanner DOMAIN (vendor/field) is
# from intensity -> compare "intensity gap" vs "anatomy gap" (AUC ~0.79-0.91).
#
# HOW TO RUN ON KAGGLE:
#   1. New Notebook -> Settings: Internet ON (required for Zenodo download),
#      Accelerator = None (CPU is enough), Persistence optional.
#   2. Paste this file into a cell (or upload as script) and Run All.
#   3. Download /kaggle/working/intensity_features.csv and
#      /kaggle/working/intensity_domain_gap.txt when done.
#
# Notes on honesty:
#   - Raw MRI intensity is NOT calibrated; absolute intensity is meaningless
#     across scanners/sequences. We therefore report BOTH:
#       (a) raw within-mask intensity stats (expected to be very domain-leaky)
#       (b) RELATIVE features (ratios / z-scored) that a fair method would use.
#   - Patient-grouped CV (no T1/T2 leakage). Deterministic seeds.
# =============================================================================
import os, sys, glob, json, subprocess, urllib.request, zipfile
import numpy as np
import pandas as pd

def sh(cmd):
    print("+", cmd); subprocess.run(cmd, shell=True, check=True)

# --- deps (Kaggle usually has numpy/pandas/sklearn; add SimpleITK) -----------
try:
    import SimpleITK as sitk
except ImportError:
    sh(f"{sys.executable} -m pip install -q SimpleITK")
    import SimpleITK as sitk

WORK = "/kaggle/working"
DATA = os.path.join(WORK, "spider")
os.makedirs(DATA, exist_ok=True)
RECORD = "10159290"

# --- 1. resolve Zenodo file list via API (robust to filename changes) --------
api = f"https://zenodo.org/api/records/{RECORD}"
with urllib.request.urlopen(api) as r:
    meta = json.load(r)
files = {f["key"]: f["links"]["self"] for f in meta["files"]}
print("Zenodo files:", {k: round(v_size/1e6,1) for k, (v_size) in
                        [(f["key"], f["size"]) for f in meta["files"]]}, "MB")

def download(key):
    if key not in files:
        raise SystemExit(f"'{key}' not in record; available={list(files)}")
    dst = os.path.join(DATA, key)
    if not os.path.exists(dst):
        print("downloading", key, "...")
        urllib.request.urlretrieve(files[key], dst)
    return dst

# images archive: pick the key containing 'image' (SPIDER: 'images.zip')
img_key = next((k for k in files if "image" in k.lower() and k.endswith(".zip")), None)
mask_key = next((k for k in files if "mask" in k.lower() and k.endswith(".zip")), None)
ov_key = next((k for k in files if k.lower().endswith("overview.csv")), None)
assert img_key and mask_key, f"could not find image/mask zips in {list(files)}"

for k in [img_key, mask_key] + ([ov_key] if ov_key else []):
    download(k)

def unzip(key, sub):
    d = os.path.join(DATA, sub)
    if not os.path.isdir(d):
        with zipfile.ZipFile(os.path.join(DATA, key)) as z:
            z.extractall(d)
    return d

img_dir = unzip(img_key, "images")
mask_dir = unzip(mask_key, "masks")
# find the folders that actually contain .mha
def find_mha_dir(root):
    for dp, _, fs in os.walk(root):
        if any(f.endswith(".mha") for f in fs):
            return dp
    return root
img_dir = find_mha_dir(img_dir); mask_dir = find_mha_dir(mask_dir)
print("images in:", img_dir, "| masks in:", mask_dir)

# --- 2. overview for vendor/field labels -------------------------------------
ov = pd.read_csv(os.path.join(DATA, ov_key)) if ov_key else \
     pd.read_csv(f"https://zenodo.org/records/{RECORD}/files/overview.csv?download=1")
ov["series"] = ov["new_file_name"].astype(str)
ov["vendor"] = ov["Manufacturer"].astype(str).str.upper().map(
    lambda m: "Siemens" if "SIEMENS" in m else ("Philips" if "PHILIPS" in m else m))
ov["field"] = ov["MagneticFieldStrength"].astype(float)
ov["patient"] = ov["series"].str.split("_").str[0]
meta = ov.set_index("series")[["vendor", "field", "patient"]]

# --- 3. intensity features within segmented structures -----------------------
VERT = list(range(1, 10)); DISC = list(range(201, 210)); CANAL = 100

def masked_stats(img_arr, mask_arr, labels):
    vals = img_arr[np.isin(mask_arr, labels)]
    if vals.size == 0:
        return dict(mean=0, std=0, p10=0, p50=0, p90=0)
    return dict(mean=float(vals.mean()), std=float(vals.std()),
                p10=float(np.percentile(vals, 10)),
                p50=float(np.percentile(vals, 50)),
                p90=float(np.percentile(vals, 90)))

rows = []
imgs = sorted(glob.glob(os.path.join(img_dir, "*.mha")))
for i, ip in enumerate(imgs):
    sid = os.path.splitext(os.path.basename(ip))[0]
    mp = os.path.join(mask_dir, sid + ".mha")
    if not os.path.exists(mp):
        continue
    try:
        ia = sitk.GetArrayFromImage(sitk.ReadImage(ip)).astype(np.float32)
        ma = sitk.GetArrayFromImage(sitk.ReadImage(mp))
        # whole-image robust scale for z-normalisation (fair, scanner-agnostic)
        gmed = np.median(ia); giqr = np.percentile(ia, 75) - np.percentile(ia, 25) + 1e-6
        v = masked_stats(ia, ma, VERT); d = masked_stats(ia, ma, DISC); c = masked_stats(ia, ma, [CANAL])
        f = {"series": sid}
        # (a) raw intensity stats
        for name, st in [("vert", v), ("disc", d), ("canal", c)]:
            for k, val in st.items():
                f[f"raw_{name}_{k}"] = val
        # (b) relative / normalised (scanner-fairer)
        f["rel_disc_vert_mean"] = d["mean"] / (v["mean"] + 1e-6)
        f["rel_canal_vert_mean"] = c["mean"] / (v["mean"] + 1e-6)
        f["z_vert_mean"] = (v["mean"] - gmed) / giqr
        f["z_disc_mean"] = (d["mean"] - gmed) / giqr
        f["vert_contrast"] = v["std"] / (v["mean"] + 1e-6)
        f["disc_contrast"] = d["std"] / (d["mean"] + 1e-6)
        if sid in meta.index:
            f["vendor"] = meta.loc[sid, "vendor"]; f["field"] = float(meta.loc[sid, "field"])
            f["patient"] = meta.loc[sid, "patient"]
        rows.append(f)
    except Exception as e:
        print("FAIL", sid, repr(e))
    if (i + 1) % 100 == 0:
        print(f"  ...{i+1}/{len(imgs)}")

idf = pd.DataFrame(rows)
idf.to_csv(os.path.join(WORK, "intensity_features.csv"), index=False)
print("wrote intensity_features.csv", idf.shape)

# --- 4. intensity domain-gap classifier (compare to anatomy gap) -------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GroupKFold

idf = idf.dropna(subset=["vendor", "field"]).reset_index(drop=True)
groups = idf["patient"].values
gkf = GroupKFold(5)
raw_cols = [c for c in idf.columns if c.startswith("raw_")]
rel_cols = [c for c in idf.columns if c.startswith(("rel_", "z_")) or c.endswith("_contrast")]

def auc(cols, tgt):
    X = idf[cols].values
    y = (idf["vendor"].values == "Siemens").astype(int) if tgt == "vendor" \
        else (idf["field"].values == 1.5).astype(int)
    clf = make_pipeline(StandardScaler(), RandomForestClassifier(
        n_estimators=400, random_state=0, class_weight="balanced", n_jobs=-1))
    return cross_val_score(clf, X, y, groups=groups, cv=gkf, scoring="roc_auc").mean()

lines = ["=== INTENSITY DOMAIN-GAP (raw SPIDER images) ==="]
lines.append(f"n={len(idf)}  raw_feats={len(raw_cols)}  rel_feats={len(rel_cols)}")
for tgt in ["vendor", "field"]:
    lines.append(f"  {tgt:<7}  RAW-intensity AUC={auc(raw_cols,tgt):.3f}   "
                 f"RELATIVE-intensity AUC={auc(rel_cols,tgt):.3f}")
lines.append("Compare with ANATOMY gap (local): vendor 0.79-0.91, field 0.82-0.89")
txt = "\n".join(lines)
print("\n" + txt)
open(os.path.join(WORK, "intensity_domain_gap.txt"), "w").write(txt)
print("\nDONE. Download intensity_features.csv + intensity_domain_gap.txt from Output.")
