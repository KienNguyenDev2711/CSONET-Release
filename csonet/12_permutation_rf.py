"""
CSoNet camera-ready revision - step 12: patient-level permutation test on the
RANDOM FOREST itself (reviewer 1, comment 5).

The submitted permutation p (0.005 = 1/201) was computed on a logistic-regression
AUC (02_domain_shift.py), while the reported AUC came from the random forest. Here
the null is built for the reported statistic: RF (400 trees, seed 0), mean of the
patient-grouped 5-fold AUCs, with labels shuffled between PATIENTS (a patient's
series keep one label). N = 1000, p = (1 + #{null >= obs}) / (N + 1) [Phipson and
Smyth 2010], so the smallest attainable p is 1/1001 = 0.001.

Robustness (the first run stalled overnight): all N permuted labelings are drawn
up front from one seeded generator, so results do not depend on scheduling; the
permutations run in parallel (one core per forest); every finished batch is
appended to a checkpoint CSV and the script resumes from it; Windows is asked not
to sleep while the script runs.
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, time, ctypes
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, cross_val_score

N = 1000; BATCH = 48; WORKERS = 12
OUTDIR = "csonet/out/revision"
CKPT = f"{OUTDIR}/permutation_rf_checkpoint.csv"

if sys.platform == "win32":   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)

df = pd.read_csv("csonet/out/features_series.csv"); df["patient"] = df["patient"].astype(str)
R3 = [c for c in df.columns if c not in ("series", "patient", "vendor", "field", "modality")]
R3_nov = [c for c in R3 if c != "voxel_volume_mm3"]
grp = df["patient"].values; upat = pd.unique(grp)

def auc(X, y, n_jobs):
    clf = make_pipeline(StandardScaler(), RandomForestClassifier(
        n_estimators=400, random_state=0, class_weight="balanced", n_jobs=n_jobs))
    return cross_val_score(clf, X, y, groups=grp, cv=GroupKFold(5), scoring="roc_auc").mean()

def permuted_labels(y):
    per = {p: y[grp == p][0] for p in upat}
    rng = np.random.default_rng(0); out = []
    for _ in range(N):
        sh = dict(zip(upat, rng.permutation([per[p] for p in upat])))
        out.append(np.array([sh[p] for p in grp]))
    return out

done = pd.read_csv(CKPT) if os.path.exists(CKPT) else pd.DataFrame(columns=["config", "i", "auc"])
lines = []
for tname, y in [("vendor", (df["vendor"] == "Siemens").astype(int).values),
                 ("field", (df["field"] == 1.5).astype(int).values)]:
    ys = permuted_labels(y)
    for fname, cols in [("R3", R3), ("R3 minus voxel", R3_nov)]:
        cfg = f"{tname}|{fname}"; X = df[cols].values
        obs = auc(X, y, -1)
        have = set(done.loc[done["config"] == cfg, "i"].astype(int))
        todo = [i for i in range(N) if i not in have]
        t = time.time()
        for b in range(0, len(todo), BATCH):
            idx = todo[b:b + BATCH]
            vals = Parallel(n_jobs=WORKERS)(delayed(auc)(X, ys[i], 1) for i in idx)
            new = pd.DataFrame({"config": cfg, "i": idx, "auc": vals})
            new.to_csv(CKPT, mode="a", header=not os.path.exists(CKPT), index=False)
            done = pd.concat([done, new], ignore_index=True)
            print(f"  {cfg}: {len(have) + b + len(idx)}/{N}  ({time.time() - t:.0f}s)", flush=True)
        null = done.loc[done["config"] == cfg, "auc"].astype(float).values
        assert len(null) == N, (cfg, len(null))
        p = (1 + np.sum(null >= obs)) / (N + 1)
        s = (f"{tname:<7} {fname:<15} RF AUC={obs:.3f}  null mean={null.mean():.3f} "
             f"95th={np.percentile(null, 95):.3f} max={null.max():.3f}  p={p:.4f}  (N={N})")
        print(s, flush=True); lines.append(s)
        open(f"{OUTDIR}/permutation_rf_results.txt", "w", encoding="utf-8").write("\n".join(lines))
