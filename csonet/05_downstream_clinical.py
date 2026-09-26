"""
CSoNet track - step 5: does the GRAPH-REL representation PRESERVE clinical signal
while being more scanner-invariant? (completes contribution #2).

Clinical targets (patient-level, from radiological_gradings.csv):
  - degeneration : any IVD with Pfirrmann grade >= 4
  - narrowing    : any IVD with Disc narrowing == 1  (geometry-grounded sign)

Two evaluations per representation (ABSOLUTE vs GRAPH-REL):
  (a) WITHIN-domain: patient-grouped 5-fold CV
  (b) CROSS-scanner: train on one vendor, test on the other (both directions)
A useful+invariant representation should retain within-domain AUC AND lose less
accuracy under the cross-scanner shift.
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.metrics import roc_auc_score

ABS = "csonet/out/features_series.csv"
GR = "csonet/out/features_graph.csv"
GRAD = "data/spider/radiological_gradings.csv"
OUTDIR = "csonet/out"

adf = pd.read_csv(ABS); gdf = pd.read_csv(GR)
for d in (adf, gdf):
    d["patient"] = d["patient"].astype(str); d["series"] = d["series"].astype(str)
m = adf.merge(gdf, on=["series", "patient", "vendor", "field"], suffixes=("_abs", "_g")).reset_index(drop=True)

# --- patient-level clinical labels ---
g = pd.read_csv(GRAD)
g.columns = [c.strip() for c in g.columns]
g = g[pd.to_numeric(g["Pfirrman grade"], errors="coerce").notna()].copy()
g["Pfirrman grade"] = g["Pfirrman grade"].astype(float)
g["Disc narrowing"] = pd.to_numeric(g["Disc narrowing"], errors="coerce").fillna(0)
lab = g.groupby("Patient").agg(
    degeneration=("Pfirrman grade", lambda s: int((s >= 4).any())),
    narrowing=("Disc narrowing", lambda s: int((s >= 1).any())),
).reset_index()
lab["patient"] = lab["Patient"].astype(str)
m = m.merge(lab[["patient", "degeneration", "narrowing"]], on="patient", how="inner").reset_index(drop=True)

id_like = ("series", "patient", "vendor", "field", "modality", "degeneration", "narrowing")
abs_cols = [c for c in adf.columns if c not in id_like]
graphrel_cols = [c for c in gdf.columns if c not in id_like and c not in ("n_nodes", "n_edges")]
REPS = {"ABSOLUTE": abs_cols, "GRAPH-REL": graphrel_cols}

lines = []
def out(s=""):
    print(s); lines.append(str(s))

def rf():
    return make_pipeline(StandardScaler(), RandomForestClassifier(
        n_estimators=400, random_state=0, class_weight="balanced", n_jobs=-1))

def within_cv(cols, y):
    return cross_val_score(rf(), m[cols].values, y, groups=m["patient"].values,
                           cv=GroupKFold(5), scoring="roc_auc").mean()

def cross_scanner(cols, y, train_vendor):
    tr = m["vendor"].values == train_vendor
    te = ~tr
    if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
        return np.nan
    clf = rf().fit(m[cols].values[tr], y[tr])
    p = clf.predict_proba(m[cols].values[te])[:, 1]
    return roc_auc_score(y[te], p)

out("=== DOWNSTREAM CLINICAL UTILITY (does GRAPH-REL keep signal?) ===")
out(f"n_series={len(m)}  n_patients={m['patient'].nunique()}")
out(f"prevalence: degeneration={m['degeneration'].mean():.2f}  narrowing={m['narrowing'].mean():.2f}")

for tgt in ["degeneration", "narrowing"]:
    y = m[tgt].values
    out(f"\n---- target: {tgt} ----")
    out(f"{'repr':<10} {'within-CV':>10} {'S->P':>8} {'P->S':>8} {'cross-mean':>11}")
    for rep, cols in REPS.items():
        w = within_cv(cols, y)
        sp = cross_scanner(cols, y, "Siemens")
        ps = cross_scanner(cols, y, "Philips")
        cm = np.nanmean([sp, ps])
        out(f"{rep:<10} {w:>10.3f} {sp:>8.3f} {ps:>8.3f} {cm:>11.3f}")

out("\nRead: high within-CV = signal present; small drop within->cross = robust to scanner shift.")
with open(os.path.join(OUTDIR, "downstream_clinical_results.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
out(f"\n[saved] {os.path.join(OUTDIR,'downstream_clinical_results.txt')}")
