"""
CSoNet camera-ready revision - step 14: 95% CIs for the downstream utility table (Table 3).
Run from the repo root. Deterministic. Writes csonet/out/revision/downstream_ci_results.txt.

Same data, features, random forest and metrics as S1b of 11_revision_analyses.py:
  within = mean over 5 seeds of per-fold AUCs (GroupKFold by patient); CI = patient-level
           bootstrap (B = 2000) of the seed-0 out-of-fold predictions (as in Table 1);
  cross  = seed-0 random forest trained on one vendor and tested on the other; the reported
           value is the mean of both directions, and its CI resamples the test patients of
           each direction independently and averages the two bootstrap AUCs.
The point estimates are recomputed and must equal revision_results.txt (S1b).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

OUT = "csonet/out"; REV = f"{OUT}/revision"
SEEDS = [0, 1, 2, 3, 4]; B = 2000
lines = []
def out(s=""):
    print(s, flush=True); lines.append(str(s))

# ------------------------------------------------------------------ data (as in step 11)
ID = ["series", "patient", "vendor", "field"]
r3 = pd.read_csv(f"{OUT}/features_series.csv")
g2 = pd.read_csv(f"{REV}/graph_full.csv")
for d in (r3, g2):
    d["series"] = d["series"].astype(str)
    if "patient" in d: d["patient"] = d["patient"].astype(str)
R3 = [c for c in r3.columns if c not in ID + ["modality"]]
SPEC = ["spec_lambda2", "spec_lambdamax", "spec_gap", "spec_entropy"]
RELA = ["relvol_mean", "relvol_std", "relvol_max", "rel_spacing_cv",
        "rel_tortuosity", "rel_canal_vol", "rel_disc_vert"]
R4 = SPEC + RELA
m = r3.merge(g2[["series"] + R4].add_suffix("_g").rename(columns={"series_g": "series"}), on="series")
R4g = [c + "_g" for c in R4]; SPECg = [c + "_g" for c in SPEC]; RELAg = [c + "_g" for c in RELA]

g = pd.read_csv("data/spider/radiological_gradings.csv"); g.columns = [c.strip() for c in g.columns]
g = g[pd.to_numeric(g["Pfirrman grade"], errors="coerce").notna()].copy()
for c in ["Pfirrman grade", "Disc narrowing"]:
    g[c] = pd.to_numeric(g[c], errors="coerce").fillna(0).astype(float)
cm = g.groupby("Patient").agg(
    degeneration=("Pfirrman grade", lambda s: int((s >= 4).any())),
    narrowing=("Disc narrowing", lambda s: int((s >= 1).any()))).reset_index()
cm["patient"] = cm["Patient"].astype(str)
m = m.merge(cm.drop(columns="Patient"), on="patient", how="left").reset_index(drop=True)
out(f"n_series={len(m)} n_patients={m.patient.nunique()}  R3={len(R3)} R4v2={len(R4)}")

# ------------------------------------------------------------------ helpers (as in step 11)
def rf(seed=0):
    return make_pipeline(StandardScaler(), RandomForestClassifier(
        n_estimators=400, random_state=seed, class_weight="balanced", n_jobs=-1))

def folds(d, y):
    return GroupKFold(5).split(d, y, d["patient"].values)

def oof(d, cols, y, seed=0):
    p = np.zeros(len(d))
    for tr, te in folds(d, y):
        p[te] = rf(seed).fit(d.iloc[tr][cols].values, y[tr]).predict_proba(d.iloc[te][cols].values)[:, 1]
    return p

def within5(d, cols, y):
    return np.mean([np.mean([roc_auc_score(y[te], rf(s).fit(d.iloc[tr][cols].values, y[tr])
                                           .predict_proba(d.iloc[te][cols].values)[:, 1])
                             for tr, te in folds(d, y)]) for s in SEEDS])

rng = np.random.default_rng(0)
def boot_sets(d, n=B):
    pat = d["patient"].values; rows = {p: np.where(pat == p)[0] for p in pd.unique(pat)}
    keys = np.array(list(rows))
    return [np.concatenate([rows[k] for k in rng.choice(keys, len(keys))]) for _ in range(n)]

def boot_aucs(y, p, sets):
    return np.array([roc_auc_score(y[s], p[s]) if len(np.unique(y[s])) == 2 else np.nan for s in sets])

# ------------------------------------------------------------------ Table 3 with CIs
S = boot_sets(m)
te_idx, te_sets = {}, {}
for trv in ["Siemens", "Philips"]:
    te = np.where((m["vendor"] != trv).values)[0]; te_idx[trv] = te
    te_sets[trv] = boot_sets(m.iloc[te].reset_index(drop=True))

for tgt in ["degeneration", "narrowing"]:
    y = m[tgt].values.astype(int)
    out(f"\n target={tgt}  (positive patients {m.groupby('patient')[tgt].first().sum()} of {m.patient.nunique()})")
    for nm, cols in [("R3", R3), ("R4v2", R4g), ("R4v2-relational", RELAg), ("R4v2-spectral", SPECg)]:
        w = within5(m, cols, y)
        p0 = oof(m, cols, y, 0)
        bw = boot_aucs(y, p0, S)
        cr, bc = [], []
        for trv in ["Siemens", "Philips"]:
            tr = (m["vendor"] == trv).values; te = te_idx[trv]
            pc = rf(0).fit(m[cols].values[tr], y[tr]).predict_proba(m[cols].values[te])[:, 1]
            cr.append(roc_auc_score(y[te], pc)); bc.append(boot_aucs(y[te], pc, te_sets[trv]))
        bcm = (bc[0] + bc[1]) / 2
        out(f"  {nm:<16} within={w:.3f} OOF(seed0)={roc_auc_score(y, p0):.3f} "
            f"95%CI[{np.nanpercentile(bw, 2.5):.4f},{np.nanpercentile(bw, 97.5):.4f}]  "
            f"S->P={cr[0]:.3f} P->S={cr[1]:.3f} cross-mean={np.mean(cr):.3f} "
            f"95%CI[{np.nanpercentile(bcm, 2.5):.4f},{np.nanpercentile(bcm, 97.5):.4f}]")

open(f"{REV}/downstream_ci_results.txt", "w").write("\n".join(lines) + "\n")
