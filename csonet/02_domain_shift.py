"""
CSoNet track - step 2 (v2): domain-shift characterization (contribution #1).
Question: can scanner DOMAIN (vendor / field) be predicted from ANATOMY-derived
geometric features (masks only, no intensity)? High predictability => domain
shift leaves a measurable fingerprint even in anatomy-derived geometry.

Fixes vs v1:
  - real multi-seed std (RF random_state varies per seed)
  - PATIENT-LEVEL permutation null (labels are patient-constant; shuffle the
    patient->label map, keep both series of a patient consistent) -> valid p
  - ABLATION: full feature set vs. feature set WITHOUT the pure-acquisition
    feature voxel_volume_mm3, to test for a residual SHAPE fingerprint beyond
    scanner resolution.
Patient-grouped CV throughout (no T1/T2 leakage). Deterministic.
"""
import sys, os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GroupKFold

FEATS = sys.argv[1] if len(sys.argv) > 1 else "csonet/out/features_series.csv"
OUTDIR = os.path.dirname(FEATS)
SEEDS = [0, 1, 2, 3, 4]
N_PERM = 200
RESOLUTION_FEATS = ["voxel_volume_mm3"]  # pure acquisition (voxel size), not anatomy

df = pd.read_csv(FEATS).reset_index(drop=True)
all_feats = [c for c in df.columns if c not in
             ("series", "patient", "vendor", "field", "modality")]
groups = df["patient"].values
gkf = GroupKFold(n_splits=5)

lines = []
def out(s=""):
    print(s); lines.append(str(s))

def cv_auc(X, y, seed):
    clf = make_pipeline(StandardScaler(), RandomForestClassifier(
        n_estimators=400, random_state=seed, class_weight="balanced", n_jobs=-1))
    return cross_val_score(clf, X, y, groups=groups, cv=gkf, scoring="roc_auc").mean()

def logreg_auc(X, y):
    clf = make_pipeline(StandardScaler(), LogisticRegression(
        max_iter=2000, class_weight="balanced"))
    return cross_val_score(clf, X, y, groups=groups, cv=gkf, scoring="roc_auc").mean()

def patient_level_permute(y, rng):
    """Shuffle labels at patient level (keep a patient's series consistent)."""
    pat = df["patient"].values
    uniq = pd.unique(pat)
    # each patient's label = its (constant) label
    lab_by_pat = {p: y[pat == p][0] for p in uniq}
    shuffled_vals = rng.permutation(list(lab_by_pat.values()))
    newmap = dict(zip(uniq, shuffled_vals))
    return np.array([newmap[p] for p in pat])

def analyze(name, y):
    out(f"\n================ target: {name} ================")
    out(f"classes: {pd.Series(y).value_counts().to_dict()}  "
        f"majority-acc={pd.Series(y).value_counts(normalize=True).max():.3f}  chance-AUC=0.500")
    for fname, feats in [("FULL features", all_feats),
                         ("NO-RESOLUTION (drop voxel_volume_mm3)",
                          [f for f in all_feats if f not in RESOLUTION_FEATS])]:
        X = df[feats].values
        rf = [cv_auc(X, y, s) for s in SEEDS]
        lr = logreg_auc(X, y)
        out(f"\n  [{fname}]  (n_feat={len(feats)})")
        out(f"    RF   ROC-AUC = {np.mean(rf):.3f} ± {np.std(rf):.3f}  (5 seeds)")
        out(f"    LogReg ROC-AUC = {lr:.3f}")
        # patient-level permutation null on LogReg (fast)
        rng = np.random.default_rng(0)
        obs = lr
        null = np.array([logreg_auc(X, patient_level_permute(y, rng)) for _ in range(N_PERM)])
        p = (np.sum(null >= obs) + 1) / (N_PERM + 1)
        out(f"    permutation null (patient-level, {N_PERM}x): "
            f"null_mean={null.mean():.3f}  null_95pct={np.percentile(null,95):.3f}  p={p:.4f}")
        # feature importance (RF on all data)
        rff = make_pipeline(StandardScaler(), RandomForestClassifier(
            n_estimators=400, random_state=0, class_weight="balanced", n_jobs=-1)).fit(X, y)
        imp = rff.named_steps["randomforestclassifier"].feature_importances_
        top = sorted(zip(feats, imp), key=lambda t: -t[1])[:6]
        out("    top features: " + ", ".join(f"{k}={v:.3f}" for k, v in top))

out("=== DOMAIN-SHIFT CHARACTERIZATION (anatomy-only, masks) ===")
out(f"n_series={len(df)}  n_patients={df['patient'].nunique()}  n_features={len(all_feats)}")

analyze("VENDOR (Siemens=1 vs Philips=0)", (df["vendor"].values == "Siemens").astype(int))
analyze("FIELD (1.5T=1 vs 3.0T=0)", (df["field"].values == 1.5).astype(int))

res_path = os.path.join(OUTDIR, "domain_shift_results.txt")
with open(res_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
out(f"\n[saved] {res_path}")
