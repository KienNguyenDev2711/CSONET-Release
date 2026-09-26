"""CSoNet camera-ready: pipeline overview figure (reviewer 2).
Drawn at the LNCS text width (12.2 cm = 4.8 in) so fonts print at their nominal size.
Vector PDF, no dash glyphs in any label."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

plt.rcParams.update({"font.family": "serif", "font.size": 6.5})
fig, ax = plt.subplots(figsize=(4.8, 2.05)); ax.set_xlim(-1, 121); ax.set_ylim(-0.5, 52); ax.axis("off")
C_IN, C_REP, C_EVAL = "#e8eef7", "#fdf1e0", "#e7f3ea"

def box(x, y, w, h, title, body, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.5",
                                fc=fc, ec="#555", lw=0.6))
    ax.text(x + w / 2, y + h - 1.2, title, ha="center", va="top", fontsize=6.4, weight="bold")
    ax.text(x + w / 2, y + (h - 5.5) / 2, body, ha="center", va="center", fontsize=5.6,
            linespacing=1.15)

def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=0.6, color="#333", mutation_scale=6))

box(0.5, 8, 17, 36, "SPIDER", "447 series\n218 patients\n4 hospitals\n\nexpert masks:\nvertebrae,\ndiscs, canal\n\nvendor, field\n(DICOM)", C_IN)
box(22, 35, 26, 15.5, "R1, R2 intensity", "signal inside masks:\nraw (R1); ratios,\nz scores (R2)", C_REP)
box(22, 17.8, 26, 15.5, "R3 mask geometry", "volumes, spacing,\nlength, counts,\nvoxel volume (mm)", C_REP)
box(22, 0.5, 26, 15.5, "R4 graph-relational", "Laplacian spectrum\nof spine graph +\nscale-free ratios", C_REP)
for y in (42.7, 25.5, 8.2):
    arrow(18.2, 26, 21.6, y)

gx = 53
ax.add_patch(FancyBboxPatch((gx - 1.5, 0.5), 17, 50, boxstyle="round,pad=0.3,rounding_size=1.5",
                            fc="white", ec="#999", lw=0.5, ls=(0, (3, 2))))
ax.text(gx + 7, 50.2, "spine graph", ha="center", va="top", fontsize=6.8, weight="bold")
ys = [9.5, 17, 24.5, 32, 39.5]
for i, y in enumerate(ys):
    dy = y - 3.75
    ax.plot([gx + 4, gx + 4], [dy, y], c="#444", lw=0.6, zorder=2)
    if i:
        ax.plot([gx + 4, gx + 4], [ys[i - 1], dy], c="#444", lw=0.6, zorder=2)
    ax.plot([gx + 4, gx + 12], [y, 24.5], c="#9aa", lw=0.5, zorder=1)
    ax.add_patch(Circle((gx + 4, dy), 1.0, fc="#e0a458", ec="k", lw=0.3, zorder=3))
    ax.add_patch(Circle((gx + 4, y), 1.8, fc="#5b7fb5", ec="k", lw=0.3, zorder=3))
    ax.text(gx + 4, y, f"V{i + 1}", ha="center", va="center", fontsize=4.6, color="w", zorder=4)
ax.add_patch(Circle((gx + 12, 24.5), 2.1, fc="#6aa56e", ec="k", lw=0.3, zorder=3))
ax.text(gx + 12, 24.5, "C", ha="center", va="center", fontsize=5.5, color="w", zorder=4)
ax.text(gx + 7, 46.5, "V vertebra, D disc,\nC canal hub", ha="center", va="top", fontsize=5.4)
arrow(48.4, 8.2, 51.2, 8.2)

box(74, 26.5, 21.5, 24, "Scanner\nprediction", "\nR1 to R4, random\nforest, patient\ngrouped CV;\nAUC, bootstrap,\npermutation", C_EVAL)
box(74, 0.5, 21.5, 24, "Controls", "voxel size,\ncoverage,\nsequence,\nfixed field,\ncase mix, ComBat", C_EVAL)
box(98, 26.5, 21.5, 24, "Utility", "degeneration,\nnarrowing;\nwithin domain\nand across\nvendors", C_EVAL)
box(98, 0.5, 21.5, 24, "Series\nnetwork", "\nk-NN graph:\nassortativity,\ncommunities,\npatient null", C_EVAL)
arrow(69.8, 30, 73.6, 38); arrow(69.8, 18, 73.6, 12)
arrow(95.9, 38, 97.6, 38); arrow(95.9, 12, 97.6, 12)
fig.subplots_adjust(0, 0, 1, 1)
fig.savefig("csonet/figures/fig0_pipeline.pdf")
fig.savefig("csonet/figures/fig0_pipeline.png", dpi=300)
print("saved fig0_pipeline")
