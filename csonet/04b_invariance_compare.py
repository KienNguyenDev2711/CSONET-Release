"""
CSoNet track - step 4b: scanner-invariance comparison (contribution #2 result).
Reads ABSOLUTE geometry (features_series.csv) and GRAPH-REL descriptors
(features_graph.csv), predicts scanner domain from each. Lower AUC = the
representation carries less scanner fingerprint = more invariant.
"""
import sys, os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GroupKFold

ABS = "csonet/out/features_series.csv"
GR = "csonet/out/features_graph.csv"
OUTDIR = "csonet/out"

adf = pd.read_csv(ABS); gdf = pd.read_csv(GR)
for d in (adf, gdf):
    d["patient"] = d["patient"].astype(str)
    d["series"] = d["series"].astype(str)

m = adf.merge(gdf, on=["series", "patient", "vendor", "field"], suffixes=("_abs", "_g"))
m = m.reset_index(drop=True)
groups = m["patient"].values
gkf = GroupKFold(5)
SEEDS = [0, 1, 2, 3, 4]

id_like = ("series", "patient", "vendor", "field", "modality")
abs_cols = [c for c in adf.columns if c not in id_like]
graph_all = [c for c in gdf.columns if c not in id_like]
graphrel_cols = [c for c in graph_all if c not in ("n_nodes", "n_edges")]  # drop raw counts

lines = []
def out(s=""):
    print(s); lines.append(str(s))

def auc_multi(cols, tgt):
    X = m[cols].values
    y = (m["vendor"].values == "Siemens").astype(int) if tgt == "vendor" \
        else (m["field"].values == 1.5).astype(int)
    aucs = []
    for s in SEEDS:
        clf = make_pipeline(StandardScaler(), RandomForestClassifier(
            n_estimators=400, random_state=s, class_weight="balanced", n_jobs=-1))
        aucs.append(cross_val_score(clf, X, y, groups=groups, cv=gkf, scoring="roc_auc").mean())
    return np.mean(aucs), np.std(aucs)

out("=== SCANNER-INVARIANCE COMPARISON (lower AUC = more scanner-invariant) ===")
out(f"merged n={len(m)}  |  ABSOLUTE feats={len(abs_cols)}  GRAPH-REL feats={len(graphrel_cols)}")
out(f"graph-rel cols: {graphrel_cols}")
for tgt in ["vendor", "field"]:
    am, asd = auc_multi(abs_cols, tgt)
    gm, gsd = auc_multi(graphrel_cols, tgt)
    out(f"\n  target={tgt}")
    out(f"    ABSOLUTE (raw physical geometry) AUC = {am:.3f} ± {asd:.3f}")
    out(f"    GRAPH-REL (spectral+relational)  AUC = {gm:.3f} ± {gsd:.3f}")
    out(f"    invariance gain (ABS - GRAPHREL) = {am-gm:+.3f}  "
        f"({'more' if gm<am else 'less'} invariant)")

with open(os.path.join(OUTDIR, "invariance_compare_results.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
out(f"\n[saved] {os.path.join(OUTDIR,'invariance_compare_results.txt')}")
