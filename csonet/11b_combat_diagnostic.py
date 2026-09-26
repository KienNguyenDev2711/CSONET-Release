"""
CSoNet camera-ready revision - step 11b: why in-fold ComBat RAISES vendor AUC on R3.
Hypothesis (tested here): batch-specific affine maps turn integer-valued coverage counts
(n_vertebrae, n_discs) and the discrete voxel volumes into non-overlapping per-batch value
lattices that a random forest reads as a batch code. Compare all R3 features vs continuous
features only, random forest vs logistic regression, raw vs ComBat (fit on training folds,
applied to the test fold with neuroCombatFromTraining).
"""
import warnings; warnings.filterwarnings("ignore")
import contextlib, io
import numpy as np, pandas as pd
from neuroCombat import neuroCombat, neuroCombatFromTraining
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

a = pd.read_csv("csonet/out/features_series.csv"); a["patient"] = a["patient"].astype(str)
R3 = [c for c in a.columns if c not in ("series", "patient", "vendor", "field", "modality")]
DISCRETE = ["voxel_volume_mm3", "n_vertebrae", "n_discs"]
CONT = [c for c in R3 if c not in DISCRETE]

def run(cols, y, harm, model):
    au = []
    for tr, te in GroupKFold(5).split(a, y, a["patient"]):
        Xtr = a.iloc[tr][cols].values.astype(float); Xte = a.iloc[te][cols].values.astype(float)
        if harm:
            with contextlib.redirect_stdout(io.StringIO()):
                f = neuroCombat(dat=Xtr.T, covars=pd.DataFrame({"b": a["vendor"].values[tr]}), batch_col="b")
                Xte = neuroCombatFromTraining(dat=Xte.T, batch=a["vendor"].values[te],
                                              estimates=f["estimates"])["data"].T
            Xtr = f["data"].T
        clf = (make_pipeline(StandardScaler(), RandomForestClassifier(
                   n_estimators=400, random_state=0, class_weight="balanced", n_jobs=-1))
               if model == "rf" else
               make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced")))
        au.append(roc_auc_score(y[te], clf.fit(Xtr, y[tr]).predict_proba(Xte)[:, 1]))
    return np.mean(au)

lines = ["=== ComBat diagnostic (batch = vendor, fitted in-fold) ==="]
for tgt, y in [("vendor", (a["vendor"] == "Siemens").astype(int).values),
               ("field", (a["field"] == 1.5).astype(int).values)]:
    for nm, cols in [("R3 all 20", R3), ("R3 continuous 17", CONT)]:
        for model in ["rf", "lr"]:
            lines.append(f"{tgt:<7} {nm:<17} {model}  raw={run(cols, y, False, model):.3f}  "
                         f"ComBat={run(cols, y, True, model):.3f}")
            print(lines[-1], flush=True)
open("csonet/out/revision/combat_diagnostic.txt", "w", encoding="utf-8").write("\n".join(lines))
