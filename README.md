# Graph-relational view of cross-scanner domain shift in lumbar spine MRI (SPIDER)

Code and derived per-series feature tables for the CSoNet 2026 short paper
"A Network and Graph-Relational View of Cross-Scanner Domain Shift in Lumbar Spine MRI"
(N. C. T. Kien, University of Information Technology, VNU-HCM).

Everything runs on a CPU. No model is trained on image volumes.

## Data (not redistributed here)

SPIDER lumbar spine MRI dataset, van der Graaf et al., Scientific Data 11, 264 (2024),
https://doi.org/10.1038/s41597-024-03090-w, Zenodo record https://doi.org/10.5281/zenodo.10159290,
licensed CC BY 4.0. Place the files as

    data/spider/masks/*.mha            (from masks.zip)
    data/spider/overview.csv
    data/spider/radiological_gradings.csv

The CSV files under `csonet/out/` are derived from SPIDER and are shared under the same
CC BY 4.0 terms, with attribution to the SPIDER authors above.

## Environment

Python 3.13, numpy, pandas, scipy, scikit-learn 1.8, networkx, SimpleITK, matplotlib,
neuroCombat (for the ComBat baseline). Run every script from the repository root.

## Order of execution

| Step | Script | Produces |
|---|---|---|
| 1 | `csonet/01_extract_anatomy_features.py data/spider/masks data/spider/overview.csv` | `out/features_series.csv` (R3) |
| 2 | `csonet/kaggle_intensity_domain_gap.py` (needs the SPIDER images) | `intensity_features.csv` (R1, R2) |
| 3 | `csonet/02_domain_shift.py` | R3 domain AUCs, voxel-size ablation |
| 4 | `csonet/03_similarity_network.py` | k-NN network metrics |
| 5 | `csonet/05_downstream_clinical.py`, `08_bootstrap_ci.py`, `09_confound_shortcut.py` | clinical utility, bootstrap, confound |
| 6 | `csonet/10_extract_revision_features.py data/spider/masks` | `out/revision/graph_full.csv` (R4, corrected graph), `graph_L5.csv`, `absolute_L5.csv` |
| 7 | `csonet/11_revision_analyses.py`, `csonet/11b_combat_diagnostic.py` | `out/revision/revision_results.txt` (all numbers in the camera-ready paper) |
| 8 | `csonet/12_permutation_rf.py` (or `csonet/kaggle_permutation_rf.py` on a Kaggle CPU notebook) | `out/revision/permutation_rf_results.txt` |
| 9 | `csonet/13_fig_pipeline.py` | Fig. 1 (pipeline) |

## Correction notice (camera-ready)

The spine graph of the submitted version (`04_graph_features.py`) left the most cranial disc
unconnected, so every graph was disconnected (lambda_2 = 0). `10_extract_revision_features.py`
builds the corrected, connected graph (disc k linked to vertebrae k minus 1 and k). All graph
numbers in the camera-ready paper come from the corrected graph. `04_graph_features.py` is kept
only so that the submitted numbers remain reproducible.
