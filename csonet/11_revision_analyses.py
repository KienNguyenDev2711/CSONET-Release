"""
CSoNet camera-ready revision - step 11: every new number requested by the reviews.
Run from the repo root after step 10. Deterministic. Writes
csonet/out/revision/revision_results.txt (the ONLY source for revised paper numbers).

S1  hierarchy + ablation with the CORRECTED graph (R4v2), bootstrap paired tests
S2  acquisition / coverage decomposition of the anatomical fingerprint (R1 comment 4)
S3  vendor vs field disentangling on the sub-cohorts that break the confound (R2)
S4  case-mix (degeneration) control of the fingerprint (R1 major 1)
S5  annotation / sequence evidence (R1 major 2)
S6  ComBat baseline, fitted inside each training fold (R2)
S7  k-NN network: same-patient edges and a patient-level permutation null
"""
import warnings; warnings.filterwarnings("ignore")
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

# ================================================================== S1
out("\n=== S1. Hierarchy and ablation with the corrected (connected) spine graph ===")
S = boot_sets(m)
res = {}
for tgt, y in [("vendor", yv_all), ("field", yf_all)]:
    out(f" target={tgt}")
    for nm, cols in [("R1 raw intensity", R1), ("R2 normalized intensity", R2),
                     ("R3 absolute geometry", R3), ("R4v2 graph-relational (spec+rel)", R4g),
                     ("R4v2 relational only", RELAg), ("R4v2 spectral only", SPECg)]:
        res[(tgt, nm)] = report(nm, m, cols, y, sets=S)
    for a_nm, b_nm in [("R3 absolute geometry", "R4v2 graph-relational (spec+rel)"),
                       ("R3 absolute geometry", "R4v2 relational only"),
                       ("R4v2 graph-relational (spec+rel)", "R4v2 relational only")]:
        md, lo, hi, pv = paired(y, res[(tgt, a_nm)][1], res[(tgt, b_nm)][1], S)
        out(f"  PAIRED {a_nm} minus {b_nm}: {md:+.3f} CI[{lo:+.3f},{hi:+.3f}] p={pv:.4f}")
    a3 = res[(tgt, "R3 absolute geometry")][0]
    for nm in ["R4v2 graph-relational (spec+rel)", "R4v2 relational only"]:
        a4 = res[(tgt, nm)][0]
        out(f"  fingerprint (AUC minus 0.5): R3 {fp(a3):.3f} -> {nm} {fp(a4):.3f}  "
            f"relative reduction {100 * (1 - fp(a4) / fp(a3)):.1f}%")

out("\n S1b. Downstream utility (clinical) for R3 vs R4v2 vs relational-only")
def cross(d, cols, y, trv, seed=0):
    tr = (d["vendor"] == trv).values; te = ~tr
    return roc_auc_score(y[te], rf(seed).fit(d[cols].values[tr], y[tr]).predict_proba(d[cols].values[te])[:, 1])
clin_oof = {}
for tgt in ["degeneration", "narrowing"]:
    y = m[tgt].values.astype(int)
    for nm, cols in [("R3", R3), ("R4v2", R4g), ("R4v2-relational", RELAg), ("R4v2-spectral", SPECg)]:
        w, _ = auc5(m, cols, y); clin_oof[(tgt, nm)] = oof(m, cols, y)
        sp, ps = cross(m, cols, y, "Siemens"), cross(m, cols, y, "Philips")
        out(f"  {tgt:<12} {nm:<16} within={w:.3f}  S->P={sp:.3f}  P->S={ps:.3f}  cross-mean={(sp + ps) / 2:.3f}")
    for a_nm, b_nm in [("R3", "R4v2"), ("R3", "R4v2-relational")]:
        md, lo, hi, pv = paired(y, clin_oof[(tgt, a_nm)], clin_oof[(tgt, b_nm)], S)
        out(f"  PAIRED within {tgt}: {a_nm} minus {b_nm} = {md:+.3f} CI[{lo:+.3f},{hi:+.3f}] p={pv:.4f}")
    for trv, tev in [("Siemens", "Philips"), ("Philips", "Siemens")]:
        tr = (m["vendor"] == trv).values; te = ~tr; mt = m[te].reset_index(drop=True)
        St = boot_sets(mt)
        pa = rf().fit(m[R3].values[tr], y[tr]).predict_proba(m[R3].values[te])[:, 1]
        pb = rf().fit(m[R4g].values[tr], y[tr]).predict_proba(m[R4g].values[te])[:, 1]
        md, lo, hi, pv = paired(y[te], pb, pa, St)
        out(f"  PAIRED cross {tgt} {trv}->{tev}: R4v2 minus R3 = {md:+.3f} CI[{lo:+.3f},{hi:+.3f}] p={pv:.4f}")

# ================================================================== S2
out("\n=== S2. Acquisition / coverage decomposition of the anatomical fingerprint ===")
m5 = m[m["spine_length_mm_a5"].notna()].reset_index(drop=True)
nosp = m5[m5["sequence"] != "t2_SPACE"].reset_index(drop=True)
for tgt, col in [("vendor", "vendor"), ("field", "field")]:
    out(f" target={tgt}")
    ysel = lambda d: ((d["vendor"] == "Siemens") if tgt == "vendor" else (d["field"] == 1.5)).astype(int).values
    report("a. R3 full field of view", m, R3, ysel(m))
    report("b. R3 minus voxel volume", m, R3_nov, ysel(m))
    report("c. fixed coverage L5 (R3 def.)", m5, R3L5a, ysel(m5))
    report("d. fixed coverage L5 minus voxel volume", m5, R3L5a_nov, ysel(m5))
    report("e. (d) excluding T2 SPACE series", nosp, R3L5a_nov, ysel(nosp))
    report("f. fixed coverage L5 graph-relational (R4)", m5, R4L5, ysel(m5))
    report("g. voxel volume ALONE", m, ["voxel_volume_mm3"], ysel(m))

# ================================================================== S3
out("\n=== S3. Disentangling vendor and field strength ===")
pv = m.groupby("patient").agg(v=("vendor", "first"), f=("field", "first"))
out("  patients by vendor x field:\n" + pd.crosstab(pv.v, pv.f).to_string())
out("  scanner models x field (series):\n" + pd.crosstab(ov["ManufacturerModelName"], ov["MagneticFieldStrength"]).to_string())
sub_v = m[m["field"] == 1.5].reset_index(drop=True)                 # vendor at fixed 1.5T
sub_f = m[m["vendor"] == "Philips"].reset_index(drop=True)          # field within Philips Ingenia
for title, d, y in [("vendor at fixed 1.5T (Siemens vs Philips Ingenia)", sub_v,
                     (sub_v["vendor"] == "Siemens").astype(int).values),
                    ("field within Philips Ingenia (1.5T vs 3T)", sub_f,
                     (sub_f["field"] == 1.5).astype(int).values)]:
    out(f" {title}")
    d5 = d[d["spine_length_mm_a5"].notna()].reset_index(drop=True)
    y5 = y[d["spine_length_mm_a5"].notna().values]
    report("R2 normalized intensity", d, R2, y, stratified=True)
    report("R3 absolute geometry", d, R3, y, stratified=True)
    report("R3 minus voxel volume", d, R3_nov, y, stratified=True)
    report("fixed coverage L5 minus voxel volume", d5, R3L5a_nov, y5, stratified=True)
    report("R4v2 graph-relational", d, R4g, y, stratified=True)

# ================================================================== S4
out("\n=== S4. Case-mix (degeneration severity) control ===")
mc = m[m[CM].notna().all(axis=1)].reset_index(drop=True)
yv = (mc["vendor"] == "Siemens").astype(int).values
out(f"  n_series with complete case-mix covariates = {len(mc)} (patients {mc.patient.nunique()}); covariates={CM}")
report("case-mix covariates ALONE -> vendor", mc, CM, yv)

def residualize(tr, te, cols):
    """Remove the linear effect of case-mix covariates, fitted on the training fold only."""
    Xtr, Xte = tr[cols].values.astype(float), te[cols].values.astype(float)
    lr = LinearRegression().fit(tr[CM].values, Xtr)
    return Xtr - lr.predict(tr[CM].values), Xte - lr.predict(te[CM].values)

for nm, cols in [("R3", R3), ("R3 minus voxel", R3_nov), ("R4v2", R4g)]:
    report(f"{nm} raw", mc, cols, yv)
    report(f"{nm} residualized on case-mix (in-fold)", mc, cols, yv, transform=residualize)
for stratum in [1, 0]:
    d = mc[mc["degeneration"] == stratum].reset_index(drop=True)
    y = (d["vendor"] == "Siemens").astype(int).values
    out(f" stratum degeneration={stratum}: patients Siemens/Philips = "
        f"{d[d.vendor == 'Siemens'].patient.nunique()}/{d[d.vendor == 'Philips'].patient.nunique()}")
    report("R3 within stratum", d, R3, y, stratified=True)
    report("R3 minus voxel within stratum", d, R3_nov, y, stratified=True)

# ================================================================== S5
out("\n=== S5. Annotation / sequence evidence ===")
both = m[m["sequence"].isin(["t1", "t2"])]
keep = both.groupby("patient")["sequence"].nunique(); keep = keep[keep == 2].index
tt = both[both["patient"].isin(keep)].reset_index(drop=True)
ys = (tt["sequence"] == "t2").astype(int).values
out(f"  patients with both T1 and T2 = {len(keep)} (same patient, visit, scanner; masks drawn on different images)")
report("sequence T2 vs T1 from R3 minus voxel", tt, R3_nov, ys)
tt5 = tt[tt["spine_length_mm_a5"].notna()].reset_index(drop=True)
report("sequence T2 vs T1 from fixed-coverage L5 minus voxel", tt5, R3L5a_nov,
       (tt5["sequence"] == "t2").astype(int).values)
report("sequence T2 vs T1 from R4v2", tt, R4g, ys)
# within-patient T1/T2 agreement of volumes (ICC-like: between-patient share of variance)
for c in ["vert_vol_mean_a5", "disc_vol_mean_a5", "spine_length_mm_a5"]:
    w = tt5.pivot_table(index="patient", columns="sequence", values=c).dropna()
    rel = (w["t2"] - w["t1"]) / ((w["t2"] + w["t1"]) / 2)
    out(f"  {c}: within-patient T2 vs T1 relative difference mean={rel.mean():+.3f} "
        f"median|.|={rel.abs().median():.3f} (n={len(w)})")
nu = m[m["umc"] == 0].reset_index(drop=True)
out(" excluding UMC (masks resampled from 3D SPACE annotations)")
report("R3 vendor, non-UMC", nu, R3, (nu["vendor"] == "Siemens").astype(int).values)
report("R3 minus voxel vendor, non-UMC", nu, R3_nov, (nu["vendor"] == "Siemens").astype(int).values)
for seq in ["t1", "t2"]:
    d = m[m["sequence"] == seq].reset_index(drop=True)
    report(f"R3 minus voxel vendor, {seq.upper()} series only", d, R3_nov,
           (d["vendor"] == "Siemens").astype(int).values)

# ================================================================== S6
out("\n=== S6. ComBat (neuroCombat, batch = vendor) fitted inside each training fold ===")
def combat(tr, te, cols):
    Xtr = tr[cols].values.astype(float); Xte = te[cols].values.astype(float)
    keep = Xtr.std(0) > 1e-12
    cv = pd.DataFrame({"batch": tr["vendor"].values})
    fit = neuroCombat(dat=Xtr[:, keep].T, covars=cv, batch_col="batch")
    ap = neuroCombatFromTraining(dat=Xte[:, keep].T, batch=te["vendor"].values, estimates=fit["estimates"])
    return fit["data"].T, ap["data"].T
for tgt, y in [("vendor", yv_all), ("field", yf_all)]:
    report(f"R3 + ComBat -> {tgt}", m, R3, y, transform=combat)
for tgt in ["degeneration", "narrowing"]:
    y = m[tgt].values.astype(int)
    report(f"R3 + ComBat -> {tgt} (within)", m, R3, y, transform=combat)

# ================================================================== S7
out("\n=== S7. k-NN similarity network: same-patient edges and permutation null ===")
Xs = StandardScaler().fit_transform(r3[R3].values)
A = kneighbors_graph(Xs, n_neighbors=8, include_self=False); A = ((A + A.T) > 0).astype(int)
G = nx.from_scipy_sparse_array(A)
pat = r3["patient"].values; ven = r3["vendor"].values; fld = r3["field"].astype(str).values
same = [(u, v) for u, v in G.edges() if pat[u] == pat[v]]
out(f"  |V|={G.number_of_nodes()} |E|={G.number_of_edges()}  same-patient edges={len(same)} "
    f"({100 * len(same) / G.number_of_edges():.1f}%)")
def assort(Gx, lab):
    nx.set_node_attributes(Gx, dict(enumerate(lab)), "a")
    return nx.attribute_assortativity_coefficient(Gx, "a")
H = G.copy(); H.remove_edges_from(same)
upat = pd.unique(pat)
for nm, lab in [("vendor", ven), ("field", fld)]:
    obs, obs_h = assort(G, lab), assort(H, lab)
    per = {p: lab[pat == p][0] for p in upat}
    r = np.random.default_rng(1); null = []
    for _ in range(1000):
        sh = dict(zip(upat, r.permutation([per[p] for p in upat])))
        null.append(assort(G, np.array([sh[p] for p in pat])))
    null = np.array(null)
    out(f"  {nm}: assortativity={obs:.3f}  without same-patient edges={obs_h:.3f}  "
        f"patient-level null mean={null.mean():.3f} 95th pct={np.percentile(null, 95):.3f}  "
        f"p={(np.sum(null >= obs) + 1) / 1001:.4f} (1000 perms)")
comms = nx.community.greedy_modularity_communities(G)
cid = np.zeros(len(r3), int)
for i, c in enumerate(comms):
    cid[list(c)] = i
out(f"  greedy modularity communities={len(comms)}  NMI(vendor)={normalized_mutual_info_score(ven, cid):.3f}")

open(f"{REV}/revision_results.txt", "w", encoding="utf-8").write("\n".join(lines))
out(f"\n[saved] {REV}/revision_results.txt")
