"""
CSoNet track - step 7: ablation of the graph representation.
Answers the reviewer question "what does the spectral part add beyond the
scale-invariant relational ratios?" by decomposing GRAPH-REL into:
  - relational-only  (ratios: relvol_*, rel_*)
  - spectral-only    (normalized-Laplacian descriptors: spec_*)
  - combined         (both = GRAPH-REL)
Lower scanner-AUC = more invariant. We report vendor & field AUC under the same
patient-grouped 5-fold CV, 5 seeds, plus the two downstream clinical targets so we
can see whether spectral trades invariance for utility.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.metrics import roc_auc_score

gdf = pd.read_csv("csonet/out/features_graph.csv")
adf = pd.read_csv("csonet/out/features_series.csv")
for d in (gdf, adf):
    d["patient"] = d["patient"].astype(str); d["series"] = d["series"].astype(str)
g = pd.read_csv("data/spider/radiological_gradings.csv")
g.columns = [c.strip() for c in g.columns]
g = g[pd.to_numeric(g["Pfirrman grade"], errors="coerce").notna()].copy()
g["Pfirrman grade"] = g["Pfirrman grade"].astype(float)
g["Disc narrowing"] = pd.to_numeric(g["Disc narrowing"], errors="coerce").fillna(0)
lab = g.groupby("Patient").agg(
    degeneration=("Pfirrman grade", lambda s: int((s >= 4).any())),
    narrowing=("Disc narrowing", lambda s: int((s >= 1).any()))).reset_index()
lab["patient"] = lab["Patient"].astype(str)
m = gdf.merge(lab[["patient", "degeneration", "narrowing"]], on="patient", how="inner").reset_index(drop=True)
groups = m["patient"].values
gkf = GroupKFold(5)
SEEDS = [0, 1, 2, 3, 4]

spectral = ["spec_lambda2", "spec_lambdamax", "spec_gap", "spec_entropy", "spec_mean"]
relational = ["relvol_mean", "relvol_std", "relvol_max", "rel_spacing_cv",
              "rel_tortuosity", "rel_canal_vol", "rel_disc_vert"]
combined = spectral + relational
SETS = {"relational-only": relational, "spectral-only": spectral, "combined (graph-rel)": combined}

def cv_auc(cols, y):
    a = []
    for s in SEEDS:
        clf = make_pipeline(StandardScaler(), RandomForestClassifier(
            n_estimators=400, random_state=s, class_weight="balanced", n_jobs=-1))
        a.append(cross_val_score(clf, m[cols].values, y, groups=groups, cv=gkf, scoring="roc_auc").mean())
    return np.mean(a), np.std(a)

def cross_scanner(cols, y, train_vendor):
    tr = m["vendor"].values == train_vendor; te = ~tr
    if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
        return np.nan
    clf = make_pipeline(StandardScaler(), RandomForestClassifier(
        n_estimators=400, random_state=0, class_weight="balanced", n_jobs=-1)).fit(m[cols].values[tr], y[tr])
    return roc_auc_score(y[te], clf.predict_proba(m[cols].values[te])[:, 1])

yv = (m["vendor"].values == "Siemens").astype(int)
yf = (m["field"].values == 1.5).astype(int)
lines = ["=== GRAPH ABLATION: spectral vs relational vs combined ===",
         f"n={len(m)}  |  spectral={len(spectral)} feats, relational={len(relational)} feats"]
lines.append(f"\n{'feature set':<22}{'vendorAUC':>10}{'fieldAUC':>10}"
             f"{'degen-within':>13}{'degen-cross':>12}{'narrow-cross':>13}")
for name, cols in SETS.items():
    v = cv_auc(cols, yv)[0]; f = cv_auc(cols, yf)[0]
    dw = cv_auc(cols, m["degeneration"].values)[0]
    dc = np.nanmean([cross_scanner(cols, m["degeneration"].values, "Siemens"),
                     cross_scanner(cols, m["degeneration"].values, "Philips")])
    nc = np.nanmean([cross_scanner(cols, m["narrowing"].values, "Siemens"),
                     cross_scanner(cols, m["narrowing"].values, "Philips")])
    lines.append(f"{name:<22}{v:>10.3f}{f:>10.3f}{dw:>13.3f}{dc:>12.3f}{nc:>13.3f}")

txt = "\n".join(lines)
print(txt)
open("csonet/out/ablation_spectral_results.txt", "w", encoding="utf-8").write(txt)
print("\n[saved] csonet/out/ablation_spectral_results.txt")
