"""
CSoNet track - step 9: the scanner CONFOUND / shortcut (answers 'so what?').
Facts (measured): (i) degeneration prevalence differs by vendor (chi2 p=0.009);
(ii) scanner is highly predictable from features (Table 1). Together these mean a
degeneration model built on scanner-leaky features can exploit scanner as a
shortcut (Glocker et al.). We quantify the SHORTCUT STRENGTH = how well scanner
identity alone predicts degeneration, with a patient-level bootstrap CI. This is a
data property; combined with each representation's scanner-AUC it bounds how much
spurious signal a model could access.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from scipy.stats import chi2_contingency

ov = pd.read_csv("data/spider/overview.csv")
ov["patient"] = ov["new_file_name"].astype(str).str.split("_").str[0]
ov["vendor"] = ov["Manufacturer"].astype(str).str.upper().map(
    lambda m: "Siemens" if "SIEMENS" in m else ("Philips" if "PHILIPS" in m else m))
pat = ov.groupby("patient").agg(vendor=("vendor","first")).reset_index()
g = pd.read_csv("data/spider/radiological_gradings.csv"); g.columns=[c.strip() for c in g.columns]
g = g[pd.to_numeric(g["Pfirrman grade"], errors="coerce").notna()].copy()
g["Pfirrman grade"]=g["Pfirrman grade"].astype(float)
lab = g.groupby("Patient").agg(degeneration=("Pfirrman grade", lambda s:int((s>=4).any()))).reset_index()
lab["patient"]=lab["Patient"].astype(str)
d = pat.merge(lab, on="patient", how="inner").reset_index(drop=True)

y = d["degeneration"].values
siem = (d["vendor"].values=="Siemens").astype(int)   # scanner indicator used as a score
lines=[]
def out(s=""): print(s); lines.append(str(s))

out("=== SCANNER CONFOUND / SHORTCUT for degeneration ===")
out(f"n_patients={len(d)}")
tab = pd.crosstab(d["vendor"], d["degeneration"])
chi2,p,_,_ = chi2_contingency(tab)
prev = d.groupby("vendor")["degeneration"].mean()
out(f"prevalence: Siemens={prev['Siemens']:.3f}  Philips={prev['Philips']:.3f}  chi2 p={p:.4f}")

# shortcut strength: scanner label as the sole predictor of degeneration
auc = roc_auc_score(y, siem)
bacc = balanced_accuracy_score(y, siem)  # predict degen iff Siemens
out(f"scanner-ALONE -> degeneration: ROC-AUC={auc:.3f}  balanced-acc={bacc:.3f}  (chance 0.5)")

# patient-level bootstrap CI for the shortcut AUC
rng=np.random.default_rng(0); B=2000; vals=[]
idx=np.arange(len(d))
for _ in range(B):
    s=rng.choice(idx,size=len(idx),replace=True)
    if len(np.unique(y[s]))<2: continue
    vals.append(roc_auc_score(y[s], siem[s]))
out(f"shortcut AUC 95% CI [{np.percentile(vals,2.5):.3f}, {np.percentile(vals,97.5):.3f}]")

out("\nInterpretation: scanner alone carries real (spurious) degeneration signal.")
out("A model on scanner-leaky features (absolute geometry: vendor AUC 0.91) can")
out("access this shortcut; the graph-relational encoding (vendor AUC 0.76,")
out("significantly lower) structurally restricts it -> the measured invariance is")
out("protective against this confound, independent of raw predictive accuracy.")
open("csonet/out/confound_shortcut_results.txt","w",encoding="utf-8").write("\n".join(lines))
out("\n[saved] csonet/out/confound_shortcut_results.txt")
