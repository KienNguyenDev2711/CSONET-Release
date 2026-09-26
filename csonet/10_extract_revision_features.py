"""
CSoNet camera-ready revision - step 10: re-extract graph features with a CORRECTED
spine graph, plus fixed-coverage variants of R3 and R4.

Why (found during camera-ready self-review, 2026-09-25):
  04_graph_features.py created one node per disc but chained only n_vert-1 discs
  (and placed disc 201 between V1 and V2, one level too high). The top disc was
  therefore always an ISOLATED node: all 447 graphs were disconnected, lambda_2 was
  0 for every series, and spec_mean / spec_entropy mostly encoded the node count and
  the number of isolated nodes (22 series with two isolated nodes were all Siemens).

Label convention (verified on the masks): vertebrae 1..9 caudal->cranial; disc 20k
lies between vertebra k-1 and vertebra k (disc 201 lies caudal to vertebra 1, whose
caudal neighbour is not labelled); 100 = spinal canal.

Corrected graph: node per present vertebra and disc + canal. Disc 20k is linked to
vertebra k-1 and vertebra k when present (a pendant node if only one is present).
Canal is linked to every vertebra. Edge weight = centroid distance / spine length
(same weighting as the submitted version). The graph is connected by construction.

Two coverages:
  full : every labelled structure in the field of view (as in the submission)
  L5   : only vertebrae 1..5 and discs 201..205, canal cropped to that axial extent,
         so every series has the same 11 nodes and the same anatomical coverage.
         Series lacking any of these labels are dropped (reported).

Outputs (csonet/out/revision/):
  graph_full.csv  graph_L5.csv  absolute_L5.csv
"""
import sys, os, glob
import numpy as np
import pandas as pd
import networkx as nx
import SimpleITK as sitk

MASK_DIR = sys.argv[1]
OVERVIEW = "data/spider/overview.csv"
OUTDIR = "csonet/out/revision"
os.makedirs(OUTDIR, exist_ok=True)

VERT = list(range(1, 10)); DISC = list(range(201, 210)); CANAL = 100


def norm_vendor(m):
    m = str(m).upper()
    return "Siemens" if "SIEMENS" in m else ("Philips" if "PHILIPS" in m else m)


def phys_coords(img, idx_zyx):
    """Physical coordinates (N,3) of voxel indices given as (z,y,x) rows."""
    ijk = idx_zyx[:, ::-1].astype(float)                     # x,y,z index
    D = np.array(img.GetDirection()).reshape(3, 3)
    sp = np.array(img.GetSpacing()); org = np.array(img.GetOrigin())
    return org + (ijk * sp) @ D.T


def spectral(G):
    L = nx.normalized_laplacian_matrix(G, weight="weight").toarray()
    ev = np.sort(np.linalg.eigvalsh(L))
    lam2, lmax = float(ev[1]), float(ev[-1])
    p = ev / ev.sum(); p = p[p > 1e-12]
    return dict(spec_lambda2=lam2, spec_lambdamax=lmax, spec_gap=lmax - lam2,
                spec_entropy=float(-(p * np.log(p)).sum()),
                spec_mean=float(ev.mean()))


def extract(fp, levels):
    img = sitk.ReadImage(fp); arr = sitk.GetArrayFromImage(img)
    sx, sy, sz = img.GetSpacing(); voxvol = float(sx * sy * sz)
    vset = VERT if levels is None else list(range(1, levels + 1))
    dset = DISC if levels is None else list(range(201, 201 + levels))
    present = set(np.unique(arr).tolist())
    if levels is not None and not (set(vset) <= present and set(dset) <= present):
        return None
    V = {}; Dd = {}
    for v in vset:
        idx = np.argwhere(arr == v)
        if len(idx):
            V[v] = (phys_coords(img, idx).mean(0), len(idx) * voxvol)
    for d in dset:
        idx = np.argwhere(arr == d)
        if len(idx):
            Dd[d] = (phys_coords(img, idx).mean(0), len(idx) * voxvol)
    if len(V) < 3:
        return None
    labs = sorted(V)
    vc = np.array([V[v][0] for v in labs]); vv = np.array([V[v][1] for v in labs])
    mu = vc.mean(0)
    _, s, vt = np.linalg.svd(vc - mu, full_matrices=False); axis = vt[0]
    if (vc[-1] - vc[0]) @ axis < 0:
        axis = -axis
    # labels are already caudal->cranial; ordering by label = anatomical order
    seg = np.linalg.norm(np.diff(vc, axis=0), axis=1)
    spine_len = float(seg.sum()); endpoint = float(np.linalg.norm(vc[-1] - vc[0]))
    dv = np.array([Dd[d][1] for d in sorted(Dd)]) if Dd else np.array([])

    # canal: full extent, or cropped to the axial span of the retained levels
    cidx = np.argwhere(arr == CANAL)
    ccent, cvol = None, 0.0
    if len(cidx):
        cp = phys_coords(img, cidx)
        if levels is not None:
            lo_pt = Dd[201][0]; hi_pt = V[levels][0]
            proj = (cp - mu) @ axis
            lo, hi = (lo_pt - mu) @ axis, (hi_pt - mu) @ axis
            keep = (proj >= min(lo, hi)) & (proj <= max(lo, hi))
            cp = cp[keep]
        if len(cp):
            ccent = cp.mean(0); cvol = len(cp) * voxvol

    # ---------------- absolute geometry (R3 definition) ----------------
    a = dict(n_vertebrae=len(V), n_discs=len(Dd), voxel_volume_mm3=voxvol,
             canal_volume_mm3=cvol,
             vert_vol_mean=vv.mean(), vert_vol_std=vv.std(), vert_vol_cv=vv.std() / vv.mean(),
             vert_vol_min=vv.min(), vert_vol_max=vv.max())
    if len(dv):
        a.update(disc_vol_mean=dv.mean(), disc_vol_std=dv.std(), disc_vol_cv=dv.std() / dv.mean())
    a.update(spine_length_mm=spine_len, intervert_spacing_mean=seg.mean(),
             intervert_spacing_std=seg.std(), intervert_spacing_cv=seg.std() / seg.mean(),
             spine_tortuosity=spine_len / endpoint,
             spine_pca_ev1=float(s[0] ** 2 / (s ** 2).sum()),
             spine_pca_ev2=float(s[1] ** 2 / (s ** 2).sum()))
    if ccent is not None:
        dl = ccent - mu
        a["canal_offset_from_axis_mm"] = float(np.linalg.norm(dl - (dl @ axis) * axis))

    # ---------------- corrected spine graph (R4) ----------------
    G = nx.Graph(); mvv = vv.mean()
    for v in labs:
        G.add_node(f"V{v}", relvol=V[v][1] / mvv)
    for d in sorted(Dd):
        G.add_node(f"D{d}", relvol=Dd[d][1] / mvv)
        k = d - 200
        for nb in (k - 1, k):
            if nb in V:
                G.add_edge(f"D{d}", f"V{nb}",
                           weight=np.linalg.norm(Dd[d][0] - V[nb][0]) / spine_len)
    if ccent is not None:
        G.add_node("C", relvol=cvol / mvv)
        for v in labs:
            G.add_edge("C", f"V{v}", weight=np.linalg.norm(ccent - V[v][0]) / spine_len)
    # vertebrae without a disc between them (missing disc label) -> direct link
    for v0, v1 in zip(labs[:-1], labs[1:]):
        if f"D{200 + v1}" not in G:
            G.add_edge(f"V{v0}", f"V{v1}", weight=np.linalg.norm(V[v1][0] - V[v0][0]) / spine_len)
    g = spectral(G)
    g["connected"] = int(nx.is_connected(G)); g["n_nodes"] = G.number_of_nodes()
    rv = np.array([d["relvol"] for _, d in G.nodes(data=True)])
    g.update(relvol_mean=rv.mean(), relvol_std=rv.std(), relvol_max=rv.max(),
             rel_spacing_cv=seg.std() / seg.mean(), rel_tortuosity=spine_len / endpoint,
             rel_canal_vol=cvol / vv.sum(),
             rel_disc_vert=(dv.mean() / mvv) if len(dv) else 0.0)
    return a, g


def main():
    ov = pd.read_csv(OVERVIEW)
    ov["series"] = ov["new_file_name"].astype(str)
    meta = {r.series: (norm_vendor(r.Manufacturer), float(r.MagneticFieldStrength))
            for r in ov.itertuples()}
    files = sorted(glob.glob(os.path.join(MASK_DIR, "*.mha")))
    out = {"graph_full": [], "graph_L5": [], "absolute_L5": []}
    dropped = []
    for i, fp in enumerate(files):
        sid = os.path.splitext(os.path.basename(fp))[0]
        ident = dict(series=sid, patient=sid.split("_")[0], vendor=meta[sid][0],
                     field=meta[sid][1],
                     sequence=sid.split("_", 1)[1])
        full = extract(fp, None)
        out["graph_full"].append({**ident, **full[1]})
        l5 = extract(fp, 5)
        if l5 is None:
            dropped.append(sid)
        else:
            out["absolute_L5"].append({**ident, **l5[0]})
            out["graph_L5"].append({**ident, **l5[1]})
        if (i + 1) % 50 == 0:
            print(f"  ...{i + 1}/{len(files)}", flush=True)
    for k, rows in out.items():
        df = pd.DataFrame(rows); df.to_csv(f"{OUTDIR}/{k}.csv", index=False)
        print(k, df.shape)
    gf = pd.DataFrame(out["graph_full"])
    print("connected graphs (full):", gf["connected"].sum(), "/", len(gf))
    print("spec_mean range (full):", gf["spec_mean"].min(), gf["spec_mean"].max())
    print("dropped from L5 coverage:", len(dropped), dropped)


if __name__ == "__main__":
    main()
