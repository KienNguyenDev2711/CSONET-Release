"""
CSoNet track - step 4: anatomical GRAPH representation (contribution #2).

For each series, build a spine graph:
  nodes  = vertebrae (ordered caudal->cranial) + IVDs + spinal canal
  edges  = anatomical adjacency: V_i -- D_i -- V_{i+1} chain, canal -- each V_i
  weights= centroid distance normalized by spine length  (scale/resolution-invariant)
  node feat = volume / mean-vertebra-volume              (scale-invariant)

Two representations compared:
  ABSOLUTE  = raw physical geometry (features_series.csv, resolution-sensitive)
  GRAPH-REL = normalized-Laplacian spectral descriptors + relational node stats
              (invariant to voxel size / global scale by construction)

Thesis test: GRAPH-REL should be LESS domain-predictable (lower vendor/field AUC
= more scanner-invariant) than ABSOLUTE, i.e. relational graph encoding removes
much of the scanner fingerprint. Reports both + writes features_graph.csv.
"""
import sys, os, glob
import numpy as np
import pandas as pd
import networkx as nx
import SimpleITK as sitk
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GroupKFold

MASK_DIR = sys.argv[1]
OVERVIEW = sys.argv[2]
ABS_FEATS = sys.argv[3] if len(sys.argv) > 3 else "csonet/out/features_series.csv"
OUT = "csonet/out/features_graph.csv"

VERT = list(range(1, 10)); DISC = list(range(201, 210)); CANAL = 100


def norm_vendor(m):
    m = str(m).upper()
    return "Siemens" if "SIEMENS" in m else ("Philips" if "PHILIPS" in m else m)


def centroid_vol(img, arr, lab, voxvol):
    idx = np.argwhere(arr == lab)
    if idx.shape[0] == 0:
        return None, 0.0
    mz = idx.mean(axis=0)
    phys = img.TransformContinuousIndexToPhysicalPoint([float(mz[2]), float(mz[1]), float(mz[0])])
    return np.array(phys), idx.shape[0] * voxvol


def spectral_summary(G):
    """Normalized-Laplacian spectral descriptors (fixed length)."""
    n = G.number_of_nodes()
    if n < 3:
        return dict(spec_lambda2=0, spec_lambdamax=0, spec_gap=0,
                    spec_entropy=0, spec_mean=0, n_nodes=n, n_edges=G.number_of_edges())
    L = nx.normalized_laplacian_matrix(G).toarray()
    ev = np.sort(np.real(np.linalg.eigvalsh(L)))
    lam2 = ev[1]            # algebraic connectivity (normalized)
    lammax = ev[-1]
    p = ev / (ev.sum() + 1e-12); p = p[p > 0]
    entropy = float(-(p * np.log(p)).sum())
    return dict(spec_lambda2=float(lam2), spec_lambdamax=float(lammax),
                spec_gap=float(lammax - lam2), spec_entropy=entropy,
                spec_mean=float(ev.mean()), n_nodes=n, n_edges=G.number_of_edges())


def build_graph_features(fp):
    img = sitk.ReadImage(fp); arr = sitk.GetArrayFromImage(img)
    sx, sy, sz = img.GetSpacing(); voxvol = sx * sy * sz

    verts = []  # (label, centroid, vol)
    for v in VERT:
        c, vol = centroid_vol(img, arr, v, voxvol)
        if c is not None:
            verts.append((v, c, vol))
    discs = []
    for d in DISC:
        c, vol = centroid_vol(img, arr, d, voxvol)
        if c is not None:
            discs.append((d, c, vol))
    ccent, cvol = centroid_vol(img, arr, CANAL, voxvol)
    if len(verts) < 3:
        return None

    vcents = np.array([v[1] for v in verts])
    vvols = np.array([v[2] for v in verts])
    mu = vcents.mean(axis=0)
    _, s, vt = np.linalg.svd(vcents - mu, full_matrices=False)
    axis = vt[0]
    vproj = (vcents - mu) @ axis
    vorder = np.argsort(vproj)
    verts_o = [verts[i] for i in vorder]
    ordered_c = np.array([v[1] for v in verts_o])
    seg = np.linalg.norm(np.diff(ordered_c, axis=0), axis=1)
    spine_len = float(seg.sum()) + 1e-9
    mean_vv = float(vvols.mean()) + 1e-9

    # order discs along same axis
    if discs:
        dproj = np.array([(d[1] - mu) @ axis for d in discs])
        discs_o = [discs[i] for i in np.argsort(dproj)]
    else:
        discs_o = []

    G = nx.Graph()
    for k, (lab, c, vol) in enumerate(verts_o):
        G.add_node(f"V{k}", relvol=vol / mean_vv)
    # place discs between consecutive vertebrae (by count)
    for j, (lab, c, vol) in enumerate(discs_o):
        G.add_node(f"D{j}", relvol=vol / mean_vv)
    # chain edges V_k - D_k - V_{k+1} (fallback: direct V-V if disc missing)
    nD = len(discs_o)
    for k in range(len(verts_o) - 1):
        w = np.linalg.norm(ordered_c[k + 1] - ordered_c[k]) / spine_len
        if k < nD:
            # split via disc node
            G.add_edge(f"V{k}", f"D{k}", weight=w / 2 + 1e-6)
            G.add_edge(f"D{k}", f"V{k+1}", weight=w / 2 + 1e-6)
        else:
            G.add_edge(f"V{k}", f"V{k+1}", weight=w + 1e-6)
    # canal node connected to each vertebra (relational hub)
    if ccent is not None:
        G.add_node("C", relvol=cvol / mean_vv)
        for k, (lab, c, vol) in enumerate(verts_o):
            w = np.linalg.norm(ccent - c) / spine_len + 1e-6
            G.add_edge("C", f"V{k}", weight=w)

    feats = spectral_summary(G)
    relvols = np.array([d["relvol"] for _, d in G.nodes(data=True)])
    feats["relvol_mean"] = float(relvols.mean())
    feats["relvol_std"] = float(relvols.std())
    feats["relvol_max"] = float(relvols.max())
    # relational geometry (scale-invariant): spacing CV, tortuosity, canal rel vol
    feats["rel_spacing_cv"] = float(seg.std() / (seg.mean() + 1e-9))
    endpoint = float(np.linalg.norm(ordered_c[-1] - ordered_c[0]))
    feats["rel_tortuosity"] = spine_len / (endpoint + 1e-9)
    feats["rel_canal_vol"] = float(cvol / (vvols.sum() + 1e-9))
    feats["rel_disc_vert"] = float((np.mean([d[2] for d in discs]) / mean_vv) if discs else 0.0)
    return feats


def main():
    ov = pd.read_csv(OVERVIEW)
    ov["series"] = ov["new_file_name"].astype(str)
    meta = {r.series: (norm_vendor(r.Manufacturer), float(r.MagneticFieldStrength),
                       r.series.split("_")[0]) for r in ov.itertuples()}
    files = sorted(glob.glob(os.path.join(MASK_DIR, "*.mha")))
    rows = []
    for i, fp in enumerate(files):
        sid = os.path.splitext(os.path.basename(fp))[0]
        try:
            f = build_graph_features(fp)
            if f is None:
                continue
            f["series"] = sid
            v, fld, pat = meta.get(sid, ("?", np.nan, sid.split("_")[0]))
            f["vendor"] = v; f["field"] = fld; f["patient"] = pat
            rows.append(f)
        except Exception as e:
            print("FAIL", sid, repr(e))
        if (i + 1) % 150 == 0:
            print(f"  ...{i+1}/{len(files)}")
    gdf = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    gdf.to_csv(OUT, index=False)
    print(f"[graph feats] wrote {OUT} shape={gdf.shape}")

    # -------- invariance comparison: ABSOLUTE vs GRAPH-REL --------
    adf = pd.read_csv(ABS_FEATS)
    m = adf.merge(gdf, on=["series", "patient", "vendor", "field"], suffixes=("_abs", ""))
    m = m.reset_index(drop=True)
    groups = m["patient"].values
    gkf = GroupKFold(5)

    abs_cols = [c for c in adf.columns if c not in ("series","patient","vendor","field","modality")]
    graph_cols = [c for c in gdf.columns if c not in ("series","patient","vendor","field")]
    # graph-rel: drop raw counts that reintroduce scale (keep spectral + rel_*)
    graphrel_cols = [c for c in graph_cols if c not in ("n_nodes","n_edges")]

    def auc(cols, target):
        X = m[cols].values
        y = (m["vendor"].values == "Siemens").astype(int) if target == "vendor" \
            else (m["field"].values == 1.5).astype(int)
        clf = make_pipeline(StandardScaler(), RandomForestClassifier(
            n_estimators=400, random_state=0, class_weight="balanced", n_jobs=-1))
        return cross_val_score(clf, X, y, groups=groups, cv=gkf, scoring="roc_auc").mean()

    print("\n=== SCANNER-INVARIANCE COMPARISON (lower AUC = more invariant) ===")
    print(f"merged n={len(m)}  abs_feats={len(abs_cols)}  graphrel_feats={len(graphrel_cols)}")
    for tgt in ["vendor", "field"]:
        a = auc(abs_cols, tgt); g = auc(graphrel_cols, tgt)
        print(f"  {tgt:<7}  ABSOLUTE AUC={a:.3f}   GRAPH-REL AUC={g:.3f}   "
              f"Δ(invariance gain)={a-g:+.3f}")
    print("  (graph-rel columns:", graphrel_cols, ")")


if __name__ == "__main__":
    main()
