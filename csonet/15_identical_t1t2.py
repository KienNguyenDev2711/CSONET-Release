"""CSoNet camera-ready, Sec. 3: patients whose T1 and T2 masks yield the same structure volume.
Reads out/features_series.csv (series '<patient>_t1' and '<patient>_t2'; T2 SPACE excluded) and
reports, per vendor, how many of the 190 patients with both series have an identical mean
vertebral volume (relative tolerance 1e-6), and, for reference, identical vertebral, disc
and canal volumes together. Also reports the Pearson r between the number of visible
vertebrae and the spine length (Sec. 5.2)."""
import numpy as np
import pandas as pd

d = pd.read_csv("csonet/out/features_series.csv")
s = d.series.str.lower()
m = d[s.str.endswith("_t1")].merge(d[s.str.endswith("_t2")], on="patient", suffixes=("_1", "_2"))
lines = [f"patients with both T1 and T2 (T2 SPACE excluded) = {len(m)}"]
for name, cols in [("mean vertebral volume", ["vert_vol_mean"]),
                   ("vertebral, disc and canal volumes", ["vert_vol_mean", "disc_vol_mean", "canal_volume_mm3"])]:
    same = np.all([np.isclose(m[c + "_1"], m[c + "_2"], rtol=1e-6, atol=0) for c in cols], axis=0)
    per = {v: f"{int(same[m.vendor_1 == v].sum())} of {int((m.vendor_1 == v).sum())}" for v in ["Philips", "Siemens"]}
    lines.append(f"identical {name}: {int(same.sum())} of {len(m)}  {per}")
lines.append(f"Pearson r(n_vertebrae, spine_length_mm) = {d.n_vertebrae.corr(d.spine_length_mm):.3f} (n={len(d)})")
open("csonet/out/revision/identical_t1t2.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
