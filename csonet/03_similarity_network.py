"""
CSoNet track - step 3: series-similarity NETWORK + community detection (contribution #3).
Build a k-NN graph over series using anatomy-derived features (physical-space
geometry). Test whether unsupervised community structure recovers scanner
domains (vendor / field) -> quantifies domain shift as a NETWORK property.

Metrics:
  - assortativity of the graph w.r.t. vendor / field labels
  - modularity of label-induced partition
  - agreement (NMI/ARI) between greedy-modularity communities and vendor/field
Network-science framing = core CSoNet Track A.
"""
import sys, os
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

FEATS = sys.argv[1] if len(sys.argv) > 1 else "csonet/out/features_series.csv"
OUTDIR = os.path.dirname(FEATS)
K = int(sys.argv[2]) if len(sys.argv) > 2 else 8

df = pd.read_csv(FEATS).reset_index(drop=True)
feat_cols = [c for c in df.columns if c not in
             ("series", "patient", "vendor", "field", "modality")]
Xs = StandardScaler().fit_transform(df[feat_cols].values)

lines = []
def out(s=""):
    print(s); lines.append(str(s))

out("=== SERIES-SIMILARITY NETWORK / community detection ===")
out(f"n_nodes={len(df)}  k={K}  n_features={len(feat_cols)}")

# kNN graph (connectivity), symmetrized
A = kneighbors_graph(Xs, n_neighbors=K, mode="connectivity", include_self=False)
A = ((A + A.T) > 0).astype(float)
G = nx.from_scipy_sparse_array(A)
vendor = df["vendor"].to_dict()
field = df["field"].astype(str).to_dict()
nx.set_node_attributes(G, vendor, "vendor")
nx.set_node_attributes(G, field, "field")
out(f"graph: |V|={G.number_of_nodes()} |E|={G.number_of_edges()} "
    f"avg_deg={2*G.number_of_edges()/G.number_of_nodes():.2f}")

# --- assortativity: do same-domain series preferentially connect? ---
for attr in ["vendor", "field"]:
    r = nx.attribute_assortativity_coefficient(G, attr)
    out(f"attribute assortativity ({attr}) = {r:.3f}  "
        f"(0=random, 1=perfectly domain-segregated)")

# --- modularity of the label-induced partition ---
def label_partition(attr):
    groups = {}
    for n, d in G.nodes(data=True):
        groups.setdefault(d[attr], set()).add(n)
    return list(groups.values())

for attr in ["vendor", "field"]:
    Q = nx.community.modularity(G, label_partition(attr))
    out(f"modularity of {attr}-partition = {Q:.3f}")

# --- unsupervised communities vs domain labels ---
comms = list(nx.community.greedy_modularity_communities(G))
comm_id = np.zeros(len(df), dtype=int)
for ci, cset in enumerate(comms):
    for n in cset:
        comm_id[n] = ci
out(f"\ngreedy-modularity communities detected = {len(comms)}")
for attr, series in [("vendor", df["vendor"].values),
                     ("field", df["field"].astype(str).values)]:
    nmi = normalized_mutual_info_score(series, comm_id)
    ari = adjusted_rand_score(series, comm_id)
    out(f"  communities vs {attr}: NMI={nmi:.3f}  ARI={ari:.3f}")

# crosstab of communities vs vendor (interpretability)
ct = pd.crosstab(pd.Series(comm_id, name="community"), df["vendor"])
out("\ncommunity x vendor crosstab:\n" + ct.to_string())

res_path = os.path.join(OUTDIR, "similarity_network_results.txt")
with open(res_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
out(f"\n[saved] {res_path}")
