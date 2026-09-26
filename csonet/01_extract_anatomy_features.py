"""
CSoNet track - step 1: extract per-series anatomical geometry features from SPIDER
segmentation masks, in PHYSICAL space (mm), robust to per-file spacing/axis order.

Label scheme (verified from data):
  0        = background
  1..9     = vertebrae (caudal->cranial; label 1 lowest)
  100      = spinal canal
  201..209 = intervertebral discs (IVDs)

Output: csonet/out/features_series.csv  (one row per series)
Each row = domain labels + flat geometric feature vector used downstream for
(1) domain-shift characterization, (2) graph representation, (3) similarity network.
"""
import sys, os, glob
import numpy as np
import pandas as pd
import SimpleITK as sitk

MASK_DIR = sys.argv[1]
OVERVIEW = sys.argv[2]
OUT      = sys.argv[3] if len(sys.argv) > 3 else "csonet/out/features_series.csv"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

VERT = list(range(1, 10))       # 1..9
DISC = list(range(201, 210))    # 201..209
CANAL = 100


def norm_vendor(m):
    m = str(m)
    if "SIEMENS" in m.upper():
        return "Siemens"
    if "PHILIPS" in m.upper():
        return "Philips"
    return m


def label_centroid_phys(img, arr, lab):
    """Physical-space centroid (x,y,z mm) and voxel count for a label."""
    idx = np.argwhere(arr == lab)          # rows of (iz,iy,ix)
    if idx.shape[0] == 0:
        return None, 0
    mean_zyx = idx.mean(axis=0)            # (iz,iy,ix)
    cont_index = [float(mean_zyx[2]), float(mean_zyx[1]), float(mean_zyx[0])]  # x,y,z
    phys = img.TransformContinuousIndexToPhysicalPoint(cont_index)
    return np.array(phys, dtype=float), int(idx.shape[0])


def extract_one(fp):
    img = sitk.ReadImage(fp)
    arr = sitk.GetArrayFromImage(img)      # (z,y,x)
    sx, sy, sz = img.GetSpacing()
    voxvol = float(sx * sy * sz)           # mm^3 per voxel

    present_vert = [v for v in VERT if np.any(arr == v)]
    present_disc = [d for d in DISC if np.any(arr == d)]

    feats = {}
    feats["n_vertebrae"] = len(present_vert)
    feats["n_discs"] = len(present_disc)
    feats["voxel_volume_mm3"] = voxvol

    # --- vertebra geometry ---
    vcents, vvols = [], []
    for v in present_vert:
        c, n = label_centroid_phys(img, arr, v)
        if c is not None:
            vcents.append(c); vvols.append(n * voxvol)
    vcents = np.array(vcents); vvols = np.array(vvols)

    # --- disc geometry ---
    dvols = []
    for d in present_disc:
        c, n = label_centroid_phys(img, arr, d)
        if c is not None:
            dvols.append(n * voxvol)
    dvols = np.array(dvols)

    # --- spinal canal ---
    ccent, cn = label_centroid_phys(img, arr, CANAL)
    feats["canal_volume_mm3"] = cn * voxvol

    # vertebra volume stats
    if len(vvols):
        feats["vert_vol_mean"] = float(vvols.mean())
        feats["vert_vol_std"]  = float(vvols.std())
        feats["vert_vol_cv"]   = float(vvols.std() / (vvols.mean() + 1e-9))
        feats["vert_vol_min"]  = float(vvols.min())
        feats["vert_vol_max"]  = float(vvols.max())
    if len(dvols):
        feats["disc_vol_mean"] = float(dvols.mean())
        feats["disc_vol_std"]  = float(dvols.std())
        feats["disc_vol_cv"]   = float(dvols.std() / (dvols.mean() + 1e-9))

    # --- spine axis via PCA on vertebra centroids (superior-inferior direction) ---
    if len(vcents) >= 3:
        mu = vcents.mean(axis=0)
        X = vcents - mu
        u, s, vt = np.linalg.svd(X, full_matrices=False)
        axis = vt[0]                       # principal (SI) axis
        proj = X @ axis                    # position of each vertebra along axis
        order = np.argsort(proj)
        ordered = vcents[order]
        seg = np.linalg.norm(np.diff(ordered, axis=0), axis=1)  # inter-vertebral spacing
        spine_len = float(seg.sum())
        endpoint = float(np.linalg.norm(ordered[-1] - ordered[0]))
        feats["spine_length_mm"] = spine_len
        feats["intervert_spacing_mean"] = float(seg.mean())
        feats["intervert_spacing_std"]  = float(seg.std())
        feats["intervert_spacing_cv"]   = float(seg.std() / (seg.mean() + 1e-9))
        # tortuosity / curvature: path length vs straight-line endpoint distance
        feats["spine_tortuosity"] = float(spine_len / (endpoint + 1e-9))
        # explained-variance ratio of 1st axis = how "straight" the column is
        feats["spine_pca_ev1"] = float((s[0] ** 2) / ((s ** 2).sum() + 1e-9))
        # lateral spread (2nd axis magnitude relative to main axis)
        feats["spine_pca_ev2"] = float((s[1] ** 2) / ((s ** 2).sum() + 1e-9)) if len(s) > 1 else 0.0
        # canal offset from vertebral column axis (mm), if canal present
        if ccent is not None:
            d_line = ccent - mu
            along = (d_line @ axis) * axis
            perp = d_line - along
            feats["canal_offset_from_axis_mm"] = float(np.linalg.norm(perp))

    return feats


def main():
    ov = pd.read_csv(OVERVIEW)
    ov["series"] = ov["new_file_name"].astype(str)
    ov["vendor"] = ov["Manufacturer"].map(norm_vendor)
    ov["field"] = ov["MagneticFieldStrength"].astype(float)
    ov["patient"] = ov["series"].str.extract(r"^(\d+)_")[0]
    ov["modality"] = np.where(ov["series"].str.contains("t1"), "T1",
                       np.where(ov["series"].str.contains("t2"), "T2", "other"))
    meta = ov.set_index("series")[["vendor", "field", "patient", "modality",
                                    "num_vertebrae", "num_discs", "PixelSpacing",
                                    "SliceThickness"]]

    files = sorted(glob.glob(os.path.join(MASK_DIR, "*.mha")))
    rows, fails = [], []
    for i, fp in enumerate(files):
        sid = os.path.splitext(os.path.basename(fp))[0]
        try:
            f = extract_one(fp)
            f["series"] = sid
            if sid in meta.index:
                m = meta.loc[sid]
                f["vendor"] = m["vendor"]; f["field"] = float(m["field"])
                f["patient"] = m["patient"]; f["modality"] = m["modality"]
            rows.append(f)
        except Exception as e:
            fails.append((sid, repr(e)))
        if (i + 1) % 100 == 0:
            print(f"  ...{i+1}/{len(files)}")

    df = pd.DataFrame(rows)
    # order columns: identifiers first
    idcols = ["series", "patient", "vendor", "field", "modality", "n_vertebrae", "n_discs"]
    other = [c for c in df.columns if c not in idcols]
    df = df[idcols + other]
    df.to_csv(OUT, index=False)
    print(f"\n[done] wrote {OUT}  shape={df.shape}  fails={len(fails)}")
    if fails:
        for s, e in fails[:10]:
            print("   FAIL", s, e)
    print("\n[vendor counts]\n", df["vendor"].value_counts(dropna=False).to_string())
    print("\n[feature columns]", [c for c in other])
    print("\n[head]\n", df.head(4).to_string())
    print("\n[NaN per column]\n", df[other].isna().sum().to_string())


if __name__ == "__main__":
    main()
