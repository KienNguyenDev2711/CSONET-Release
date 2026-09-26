"""
CSoNet track - step 8: patient-level bootstrap 95% CIs + paired significance tests.

Rationale (reviewer point): the AUC comparisons need uncertainty. We use
out-of-fold (OOF) predicted probabilities from patient-grouped 5-fold CV (fixed
seed), then bootstrap by resampling PATIENTS with replacement (B=2000), so both
T1/T2 of a patient move together. For a paired model comparison we recompute both
models' AUC on the SAME resampled patients and take the difference -> CI + a
bootstrap two-sided p-value (2*min(P(d<=0),P(d>=0))).

Cross-scanner: train on one vendor, bootstrap the held-out vendor's TEST patients.

Deterministic (np.random.default_rng(0)). Only measured numbers are printed.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.metrics import roc_auc_score

B = 2000
rng = np.random.default_rng(0)

# ---------- load & merge (same as step 6/5) ----------
abs_df = pd.read_csv("csonet/out/features_series.csv")
gr_df = pd.read_csv("csonet/out/features_graph.csv")
it_df = pd.read_csv("csonet/intensity_features.csv")
for d in (abs_df, gr_df, it_df):
    d["series"] = d["series"].astype(str)
    if "patient" in d: d["patient"] = d["patient"].astype(str)
m = abs_df.merge(gr_df, on=["series","patient","vendor","field"], suffixes=("_abs","_g")) \
          .merge(it_df.drop(columns=[c for c in ("vendor","field","patient") if c in it_df]), on="series") \
          .reset_index(drop=True)
g = pd.read_csv("data/spider/radiological_gradings.csv"); g.columns=[c.strip() for c in g.columns]
g = g[pd.to_numeric(g["Pfirrman grade"], errors="coerce").notna()].copy()
g["Pfirrman grade"]=g["Pfirrman grade"].astype(float)
g["Disc narrowing"]=pd.to_numeric(g["Disc narrowing"], errors="coerce").fillna(0)
lab=g.groupby("Patient").agg(degeneration=("Pfirrman grade", lambda s:int((s>=4).any())),
                             narrowing=("Disc narrowing", lambda s:int((s>=1).any()))).reset_index()
lab["patient"]=lab["Patient"].astype(str)
m=m.merge(lab[["patient","degeneration","narrowing"]], on="patient", how="left")

idl=("series","patient","vendor","field","modality","degeneration","narrowing")
anat_abs=[c for c in abs_df.columns if c not in idl]
anat_gr=[c for c in gr_df.columns if c not in idl and c not in ("n_nodes","n_edges")]
int_raw=[c for c in it_df.columns if c.startswith("raw_")]
int_rel=[c for c in it_df.columns if c.startswith(("rel_","z_")) or c.endswith("_contrast")]
patients=m["patient"].values
pat_to_rows={p:np.where(patients==p)[0] for p in pd.unique(patients)}
uniq_pat=np.array(list(pat_to_rows.keys()))

def rf(): return make_pipeline(StandardScaler(), RandomForestClassifier(
    n_estimators=400, random_state=0, class_weight="balanced", n_jobs=-1))

def oof(cols, y, groups):
    return cross_val_predict(rf(), m.loc[:,cols].values, y, groups=groups,
                             cv=GroupKFold(5), method="predict_proba")[:,1]

def boot_idx():
    samp=rng.choice(uniq_pat, size=len(uniq_pat), replace=True)
    return np.concatenate([pat_to_rows[p] for p in samp])

def ci_auc(y, p, idxs):
    vals=[]
    for idx in idxs:
        yy=y[idx]
        if len(np.unique(yy))<2: continue
        vals.append(roc_auc_score(yy, p[idx]))
    return np.percentile(vals,2.5), np.percentile(vals,97.5)

def paired_diff(y, pa, pb, idxs):
    """diff = AUC(pa) - AUC(pb) per resample -> CI + two-sided p."""
    d=[]
    for idx in idxs:
        yy=y[idx]
        if len(np.unique(yy))<2: continue
        d.append(roc_auc_score(yy,pa[idx])-roc_auc_score(yy,pb[idx]))
    d=np.array(d)
    p=2*min((d<=0).mean(),(d>=0).mean())
    return d.mean(), np.percentile(d,2.5), np.percentile(d,97.5), p

lines=[]
def out(s=""): print(s); lines.append(str(s))

idxs=[boot_idx() for _ in range(B)]   # shared resamples for paired tests
out(f"=== PATIENT-LEVEL BOOTSTRAP (B={B}, OOF from grouped 5-fold, seed=0) ===")
out(f"n_series={len(m)} n_patients={len(uniq_pat)}")

# ---------- A. domain-gap hierarchy ----------
groups=patients
for dom,yfun in [("vendor", (m["vendor"].values=="Siemens").astype(int)),
                 ("field", (m["field"].values==1.5).astype(int))]:
    out(f"\n--- DOMAIN = {dom} : per-representation OOF AUC [95% CI] ---")
    probs={}
    for name,cols in [("intensity_raw",int_raw),("intensity_rel",int_rel),
                      ("anatomy_abs",anat_abs),("anatomy_graphrel",anat_gr)]:
        p=oof(cols,yfun,groups); probs[name]=p
        pt=roc_auc_score(yfun,p); lo,hi=ci_auc(yfun,p,idxs)
        out(f"  {name:<18} AUC={pt:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    md,lo,hi,pv=paired_diff(yfun, probs["anatomy_abs"], probs["anatomy_graphrel"], idxs)
    out(f"  DIFF absolute - graphrel = {md:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  p={pv:.4f}"
        f"  ({'SIGNIFICANT' if (lo>0 or hi<0) else 'n.s.'})")

# ---------- B. clinical within-domain: abs vs graphrel ----------
for tgt in ["degeneration","narrowing"]:
    y=m[tgt].values
    out(f"\n--- CLINICAL(within) = {tgt} : abs vs graphrel ---")
    pa=oof(anat_abs,y,groups); pb=oof(anat_gr,y,groups)
    for nm,p in [("absolute",pa),("graphrel",pb)]:
        pt=roc_auc_score(y,p); lo,hi=ci_auc(y,p,idxs)
        out(f"  {nm:<10} AUC={pt:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    md,lo,hi,pv=paired_diff(y,pa,pb,idxs)
    out(f"  DIFF absolute - graphrel = {md:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  p={pv:.4f}"
        f"  ({'SIGNIFICANT' if (lo>0 or hi<0) else 'n.s.'})")

# ---------- C. clinical cross-scanner: bootstrap test vendor ----------
for tgt in ["degeneration","narrowing"]:
    for train_v,test_v in [("Siemens","Philips"),("Philips","Siemens")]:
        y=m[tgt].values; tr=m["vendor"].values==train_v; te=~tr
        if len(np.unique(y[tr]))<2: continue
        pa=rf().fit(m.loc[tr,anat_abs].values,y[tr]).predict_proba(m.loc[te,anat_abs].values)[:,1]
        pb=rf().fit(m.loc[tr,anat_gr].values,y[tr]).predict_proba(m.loc[te,anat_gr].values)[:,1]
        yte=y[te]; te_pat=patients[te]
        tp={p:np.where(te_pat==p)[0] for p in pd.unique(te_pat)}
        tup=np.array(list(tp.keys()))
        tidx=[np.concatenate([tp[p] for p in rng.choice(tup,size=len(tup),replace=True)]) for _ in range(B)]
        a_pt=roc_auc_score(yte,pa); b_pt=roc_auc_score(yte,pb)
        alo,ahi=ci_auc(yte,pa,tidx); blo,bhi=ci_auc(yte,pb,tidx)
        md,lo,hi,pv=paired_diff(yte,pb,pa,tidx)  # graphrel - absolute
        out(f"\n--- CROSS {tgt}: train {train_v} -> test {test_v} (n_test_pat={len(tup)}) ---")
        out(f"  absolute AUC={a_pt:.3f} CI[{alo:.3f},{ahi:.3f}] | graphrel AUC={b_pt:.3f} CI[{blo:.3f},{bhi:.3f}]")
        out(f"  DIFF graphrel - absolute = {md:+.3f} CI[{lo:+.3f},{hi:+.3f}] p={pv:.4f}"
            f"  ({'SIGNIFICANT' if (lo>0 or hi<0) else 'n.s.'})")

open("csonet/out/bootstrap_ci_results.txt","w",encoding="utf-8").write("\n".join(lines))
out("\n[saved] csonet/out/bootstrap_ci_results.txt")
