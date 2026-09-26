"""
CSoNet camera-ready revision - step 11c: two extra tests requested by the self-review.
Run from the repo root after step 11. Deterministic. Writes
csonet/out/revision/revision_extra_results.txt.

X0  sanity check: reproduce the S1 paired test R3 minus R4v2 (vendor +0.153) with the
    identical bootstrap sets, so the new paired tests below use the same resamples.
X1  fair baseline for the graph encoding: R3 WITHOUT voxel volume vs R4v2
    (full coverage, fixed coverage, and the two sub-cohorts that break the confound).
X2  ComBat on the 17 continuous R3 features under the main protocol (5 RF seeds,
    bootstrap CI, paired tests), for vendor AND field strength.
Data loading and helpers are copied verbatim from 11_revision_analyses.py.
"""
import warnings; warnings.filterwarnings("ignore")
import contextlib, io
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, normalized_mutual_info_score
from sklearn.neighbors import kneighbors_graph
from neuroCombat import neuroCombat, neuroCombatFromTraining

OUT = "csonet/out"; REV = f"{OUT}/revision"
SEEDS = [0, 1, 2, 3, 4]; B = 2000
lines = []
def out(s=""):
    print(s, flush=True); lines.append(str(s))

# ------------------------------------------------------------------ data
ID = ["series", "patient", "vendor", "field"]
r3 = pd.read_csv(f"{OUT}/features_series.csv")
it = pd.read_csv("csonet/intensity_features.csv")
g2 = pd.read_csv(f"{REV}/graph_full.csv")
a5 = pd.read_csv(f"{REV}/absolute_L5.csv")
g5 = pd.read_csv(f"{REV}/graph_L5.csv")
for d in (r3, it, g2, a5, g5):
    d["series"] = d["series"].astype(str)
    if "patient" in d: d["patient"] = d["patient"].astype(str)

R3 = [c for c in r3.columns if c not in ID + ["modality"]]
R1 = [c for c in it.columns if c.startswith("raw_")]
R2 = [c for c in it.columns if c.startswith(("rel_", "z_")) or c.endswith("_contrast")]
SPEC = ["spec_lambda2", "spec_lambdamax", "spec_gap", "spec_entropy"]   # spec_mean == 1
RELA = ["relvol_mean", "relvol_std", "relvol_max", "rel_spacing_cv",
        "rel_tortuosity", "rel_canal_vol", "rel_disc_vert"]
R4 = SPEC + RELA
R3L5 = [c for c in a5.columns if c not in ID + ["sequence", "n_vertebrae", "n_discs"]]
R3L5_nov = [c for c in R3L5 if c != "voxel_volume_mm3"]
R3_nov = [c for c in R3 if c != "voxel_volume_mm3"]

m = (r3.merge(it.drop(columns=[c for c in ("vendor", "field", "patient") if c in it]), on="series")
       .merge(g2[["series", "sequence"] + R4].add_suffix("_g").rename(
            columns={"series_g": "series", "sequence_g": "sequence"}), on="series"))
R4g = [c + "_g" for c in R4]; SPECg = [c + "_g" for c in SPEC]; RELAg = [c + "_g" for c in RELA]
L5 = a5.merge(g5[["series"] + R4].add_suffix("_g5").rename(columns={"series_g5": "series"}), on="series")
R4L5 = [c + "_g5" for c in R4]
m = m.merge(L5[["series"] + R3L5 + R4L5].rename(columns={c: c + "_a5" for c in R3L5}),
            on="series", how="left")
R3L5a = [c + "_a5" for c in R3L5]; R3L5a_nov = [c + "_a5" for c in R3L5_nov]

# clinical + case-mix covariates (patient level)
g = pd.read_csv("data/spider/radiological_gradings.csv"); g.columns = [c.strip() for c in g.columns]
g = g[pd.to_numeric(g["Pfirrman grade"], errors="coerce").notna()].copy()
for c in ["Pfirrman grade", "Modic", "UP endplate", "LOW endplate", "Spondylolisthesis",
          "Disc herniation", "Disc narrowing", "Disc bulging"]:
    g[c] = pd.to_numeric(g[c], errors="coerce").fillna(0).astype(float)
cm = g.groupby("Patient").agg(
    degeneration=("Pfirrman grade", lambda s: int((s >= 4).any())),
    narrowing=("Disc narrowing", lambda s: int((s >= 1).any())),
    cm_pfirr_mean=("Pfirrman grade", "mean"), cm_pfirr_max=("Pfirrman grade", "max"),
    cm_frac_pf4=("Pfirrman grade", lambda s: (s >= 4).mean()),
    cm_frac_narrow=("Disc narrowing", lambda s: (s >= 1).mean()),
    cm_frac_hern=("Disc herniation", "mean"), cm_frac_bulge=("Disc bulging", "mean"),
    cm_any_listh=("Spondylolisthesis", "max"), cm_frac_modic=("Modic", lambda s: (s > 0).mean()),
    cm_frac_endpl=("UP endplate", "mean")).reset_index()
cm["patient"] = cm["Patient"].astype(str)
ov = pd.read_csv("data/spider/overview.csv")
ov["patient"] = ov["new_file_name"].astype(str).str.split("_").str[0]
sex = ov.groupby("patient")["sex"].first().str.strip().map({"F": 1, "M": 0}).rename("cm_female")
m = m.merge(cm.drop(columns="Patient"), on="patient", how="left").merge(sex, on="patient", how="left")
CM = [c for c in m.columns if c.startswith("cm_")]
m = m.reset_index(drop=True)
# UMC = the only hospital with T2 SPACE (SPIDER descriptor, Data collection section)
umc = set(m.loc[m["sequence"] == "t2_SPACE", "patient"])
m["umc"] = m["patient"].isin(umc).astype(int)
yv_all = (m["vendor"] == "Siemens").astype(int).values
yf_all = (m["field"] == 1.5).astype(int).values
out(f"merged n_series={len(m)} n_patients={m.patient.nunique()}  "
    f"R1={len(R1)} R2={len(R2)} R3={len(R3)} R4v2={len(R4)} R3L5={len(R3L5)} case-mix={len(CM)}")
out(f"UMC patients (have T2 SPACE) = {len(umc)}; vendors: "
    f"{m[m.umc == 1].groupby('vendor').patient.nunique().to_dict()}")

# ------------------------------------------------------------------ helpers
def rf(seed=0):
    return make_pipeline(StandardScaler(), RandomForestClassifier(
        n_estimators=400, random_state=seed, class_weight="balanced", n_jobs=-1))

def splitter(stratified, seed):
    return StratifiedGroupKFold(5, shuffle=True, random_state=seed) if stratified else GroupKFold(5)

def oof(d, cols, y, seed=0, stratified=False, transform=None):
    """Out-of-fold P(y=1). transform(train_df, test_df) -> (Xtr, Xte) for in-fold harmonization."""
    p = np.zeros(len(d)); grp = d["patient"].values
    for tr, te in splitter(stratified, seed).split(d, y, grp):
        if transform is None:
            Xtr, Xte = d.iloc[tr][cols].values, d.iloc[te][cols].values
        else:
            Xtr, Xte = transform(d.iloc[tr], d.iloc[te], cols)
        p[te] = rf(seed).fit(Xtr, y[tr]).predict_proba(Xte)[:, 1]
    return p

def fold_auc(d, cols, y, seed, stratified=False, transform=None):
    """Mean of per-fold AUCs (the metric of the submitted Table 1)."""
    a = []; grp = d["patient"].values
    for tr, te in splitter(stratified, seed).split(d, y, grp):
        if transform is None:
            Xtr, Xte = d.iloc[tr][cols].values, d.iloc[te][cols].values
        else:
            Xtr, Xte = transform(d.iloc[tr], d.iloc[te], cols)
        p = rf(seed).fit(Xtr, y[tr]).predict_proba(Xte)[:, 1]
        a.append(roc_auc_score(y[te], p))
    return np.mean(a)

def auc5(d, cols, y, stratified=False, transform=None):
    v = [fold_auc(d, cols, y, s, stratified, transform) for s in SEEDS]
    return np.mean(v), np.std(v)

rng = np.random.default_rng(0)
def boot_sets(d, n=B):
    pat = d["patient"].values; rows = {p: np.where(pat == p)[0] for p in pd.unique(pat)}
    keys = np.array(list(rows))
    return [np.concatenate([rows[k] for k in rng.choice(keys, len(keys))]) for _ in range(n)]

def ci(y, p, sets):
    v = [roc_auc_score(y[s], p[s]) for s in sets if len(np.unique(y[s])) == 2]
    return np.percentile(v, 2.5), np.percentile(v, 97.5)

def paired(y, pa, pb, sets):
    d = np.array([roc_auc_score(y[s], pa[s]) - roc_auc_score(y[s], pb[s])
                  for s in sets if len(np.unique(y[s])) == 2])
    return d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5), 2 * min((d <= 0).mean(), (d >= 0).mean())

def fp(a):  # fingerprint magnitude above chance
    return a - 0.5

def report(name, d, cols, y, stratified=False, transform=None, sets=None):
    mu, sd = auc5(d, cols, y, stratified, transform)
    p = oof(d, cols, y, 0, stratified, transform)
    lo, hi = ci(y, p, sets if sets is not None else boot_sets(d))
    out(f"  {name:<46} AUC={mu:.3f}+-{sd:.3f}  OOF(seed0)={roc_auc_score(y, p):.3f} "
        f"95%CI[{lo:.3f},{hi:.3f}]  n_ser={len(d)} n_pat={d.patient.nunique()}")
    return mu, p

def pair_out(label, y, pa, pb, sets):
    md, lo, hi, pv = paired(y, pa, pb, sets)
    out(f"  PAIRED {label}: {md:+.3f} CI[{lo:+.3f},{hi:+.3f}] p={pv:.4f}")

# ================================================================== X0
out("=== X0. Sanity check against S1 (must equal revision_results.txt) ===")
S = boot_sets(m)                      # first rng call, identical to S1 in step 11
P = {}
for tgt, y in [("vendor", yv_all), ("field", yf_all)]:
    out(f" target={tgt}")
    P[(tgt, "R3")] = report("R3 absolute geometry", m, R3, y, sets=S)
    P[(tgt, "R4")] = report("R4v2 graph-relational (spec+rel)", m, R4g, y, sets=S)
    pair_out("R3 minus R4v2", y, P[(tgt, "R3")][1], P[(tgt, "R4")][1], S)

# ================================================================== X1
out("\n=== X1. Fair baseline: R3 without voxel volume vs the graph-relational encoding ===")
for tgt, y in [("vendor", yv_all), ("field", yf_all)]:
    out(f" target={tgt} (full coverage, 447 series)")
    P[(tgt, "R3nov")] = report("R3 minus voxel volume", m, R3_nov, y, sets=S)
    pair_out("R3 minus voxel minus R4v2", y, P[(tgt, "R3nov")][1], P[(tgt, "R4")][1], S)
    a3, a3n, a4 = P[(tgt, "R3")][0], P[(tgt, "R3nov")][0], P[(tgt, "R4")][0]
    out(f"  fingerprint (AUC minus 0.5): R3 {fp(a3):.3f} -> R3 minus voxel {fp(a3n):.3f} "
        f"({100 * (1 - fp(a3n) / fp(a3)):.1f}% by dropping one feature) -> R4v2 {fp(a4):.3f} "
        f"({100 * (1 - fp(a4) / fp(a3n)):.1f}% further)")
m5 = m[m["spine_length_mm_a5"].notna()].reset_index(drop=True)
S5 = boot_sets(m5)
for tgt in ["vendor", "field"]:
    y = ((m5["vendor"] == "Siemens") if tgt == "vendor" else (m5["field"] == 1.5)).astype(int).values
    out(f" target={tgt} (fixed coverage L5, 446 series)")
    _, pa = report("fixed coverage L5 minus voxel volume", m5, R3L5a_nov, y, sets=S5)
    _, pb = report("fixed coverage L5 graph-relational (R4)", m5, R4L5, y, sets=S5)
    pair_out("L5 minus voxel minus L5 R4", y, pa, pb, S5)
sub_v = m[m["field"] == 1.5].reset_index(drop=True)
sub_f = m[m["vendor"] == "Philips"].reset_index(drop=True)
for title, d, y in [("vendor at fixed 1.5T", sub_v, (sub_v["vendor"] == "Siemens").astype(int).values),
                    ("field within Philips Ingenia", sub_f, (sub_f["field"] == 1.5).astype(int).values)]:
    out(f" {title}")
    Sd = boot_sets(d)
    _, pa = report("R3 minus voxel volume", d, R3_nov, y, stratified=True, sets=Sd)
    _, pb = report("R4v2 graph-relational", d, R4g, y, stratified=True, sets=Sd)
    pair_out("R3 minus voxel minus R4v2", y, pa, pb, Sd)

# ================================================================== X2
out("\n=== X2. ComBat (batch = vendor, in-fold) on the 17 continuous R3 features, main protocol ===")
DISCRETE = ["voxel_volume_mm3", "n_vertebrae", "n_discs"]
CONT = [c for c in R3 if c not in DISCRETE]
out(f"  continuous features = {len(CONT)}")
def combat(tr, te, cols):
    Xtr = tr[cols].values.astype(float); Xte = te[cols].values.astype(float)
    keep = Xtr.std(0) > 1e-12
    cv = pd.DataFrame({"batch": tr["vendor"].values})
    with contextlib.redirect_stdout(io.StringIO()):
        fit = neuroCombat(dat=Xtr[:, keep].T, covars=cv, batch_col="batch")
        ap = neuroCombatFromTraining(dat=Xte[:, keep].T, batch=te["vendor"].values, estimates=fit["estimates"])
    return fit["data"].T, ap["data"].T
for tgt, y in [("vendor", yv_all), ("field", yf_all)]:
    out(f" target={tgt}")
    _, pr = report("R3 continuous, raw", m, CONT, y, sets=S)
    _, pc = report("R3 continuous + ComBat", m, CONT, y, transform=combat, sets=S)
    pair_out("raw minus ComBat", y, pr, pc, S)
    pair_out("ComBat minus R4v2", y, pc, P[(tgt, "R4")][1], S)
for tgt in ["degeneration", "narrowing"]:
    y = m[tgt].values.astype(int)
    report(f"R3 continuous + ComBat -> {tgt} (within)", m, CONT, y, transform=combat, sets=S)

open(f"{REV}/revision_extra_results.txt", "w", encoding="utf-8").write("\n".join(lines))
out(f"\n[saved] {REV}/revision_extra_results.txt")
