"""CSoNet camera-ready: study-design overview figure (reviewer 2).
Drawn at the LNCS text width (12.2 cm = 4.8 in) so fonts print at their nominal size.
Data flow: SPIDER -> four representations -> ONE arrow into a group of four PARALLEL
analyses (no arrows between analyses, since none feeds another). The spine graph is a
detail of R4 and is linked to it by a dashed line, not by a data-flow arrow.
Vector PDF, no dash glyphs in any label."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse


def Circle(xy, r, **kw):
    """Round node in data units despite unequal x/y scaling of the axes."""
    k = ((ax.get_ylim()[1] - ax.get_ylim()[0]) / fig.get_figheight()) \
        / ((ax.get_xlim()[1] - ax.get_xlim()[0]) / fig.get_figwidth())
    return Ellipse(xy, 2 * r, 2 * r * k, **kw)

plt.rcParams.update({"font.family": "serif", "font.size": 6.5})
fig, ax = plt.subplots(figsize=(4.8, 1.85)); ax.set_xlim(-1, 121); ax.set_ylim(-0.8, 53); ax.axis("off")
C_IN, C_REP, C_EVAL, C_GRP = "#e8eef7", "#fdf1e0", "#e7f3ea", "#f7faf7"

def box(x, y, w, h, title, body, fc, tfs=6.3, bfs=5.6, lw=0.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.5",
                                fc=fc, ec="#555", lw=lw))
    ax.text(x + w / 2, y + h - 1.0, title, ha="center", va="top", fontsize=tfs, weight="bold")
    ax.text(x + w / 2, y + (h - 5.0) / 2, body, ha="center", va="center", fontsize=bfs,
            linespacing=1.15)

def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#333", mutation_scale=7,
                                shrinkA=0, shrinkB=0))

# input and representations
box(0.5, 9, 15.5, 34, "SPIDER", "447 series\n218 patients\n4 hospitals\n\nexpert masks:\nvertebrae,\ndiscs, canal\n\nvendor, field\n(DICOM)", C_IN)
reps = [(38.5, "R1, R2 intensity", "signal inside masks:\nraw (R1); ratios,\nz scores (R2)"),
        (25.5, "R3 mask geometry", "volumes, spacing,\nlength, counts,\nvoxel volume (mm)"),
        (12.5, "R4 graph-relational", "Laplacian spectrum\nof spine graph +\nscale-free ratios")]
for y, t, b in reps:
    box(20, y, 25, 11.5, t, b, C_REP)
for y, _, _ in reps:
    arrow(16.4, 26, 19.6, y + 5.75)

# collector: all representations -> analyses (single data-flow arrow)
xb = 47.5
for y, _, _ in reps:
    ax.plot([45.4, xb], [y + 5.75, y + 5.75], c="#333", lw=0.7)
ax.plot([xb, xb], [18.25, 44.25], c="#333", lw=0.7)
arrow(xb, 31.25, 63.6, 31.25)

# spine graph: detail of R4 (dashed link, no arrow)
gx, gy0 = 23, -0.3
ax.add_patch(FancyBboxPatch((gx - 2.5, gy0), 30, 10.2, boxstyle="round,pad=0.3,rounding_size=1.2",
                            fc="white", ec="#999", lw=0.5, ls=(0, (3, 2))))
ax.plot([32.5, 32.5], [12.4, 10.3], c="#999", lw=0.5, ls=(0, (2, 1.5)))
xs = [gx + 2 + 5.2 * i for i in range(4)]; yv = gy0 + 6.3
for i, x in enumerate(xs):
    if i:
        ax.plot([xs[i - 1], x], [yv, yv], c="#444", lw=0.6, zorder=2)
        ax.add_patch(Circle(((xs[i - 1] + x) / 2, yv), 0.8, fc="#e0a458", ec="k", lw=0.3, zorder=3))
    ax.plot([x, gx + 10], [yv, gy0 + 1.9], c="#9aa", lw=0.5, zorder=1)
    ax.add_patch(Circle((x, yv), 1.5, fc="#5b7fb5", ec="k", lw=0.3, zorder=3))
    ax.text(x, yv, f"V{i + 1}", ha="center", va="center", fontsize=4.2, color="w", zorder=4)
ax.add_patch(Circle((gx + 10, gy0 + 1.9), 1.5, fc="#6aa56e", ec="k", lw=0.3, zorder=3))
ax.text(gx + 10, gy0 + 1.9, "C", ha="center", va="center", fontsize=4.6, color="w", zorder=4)
ax.text(gx + 25.5, gy0 + 5.1, "V vertebra\nD disc\nC canal", ha="center", va="center",
        fontsize=4.8, linespacing=1.1)

# analyses group: four parallel analyses, no arrows between them
ax.add_patch(FancyBboxPatch((63.9, -0.3), 56.6, 53.0, boxstyle="round,pad=0.3,rounding_size=1.8",
                            fc=C_GRP, ec="#888", lw=0.6))
ax.text(92.2, 52.4, "Parallel analyses (same protocol)", ha="center", va="top",
        fontsize=6.3, weight="bold")
box(66, 25.5, 25.5, 22.5, "Scanner\npredictability", "random forest,\npatient grouped CV;\nAUC, bootstrap,\npermutation", C_EVAL)
box(93.5, 25.5, 25.5, 22.5, "Acquisition\ncontrols", "voxel size, coverage,\nsequence, fixed field,\ncase mix, UMC,\nComBat", C_EVAL)
box(66, 0.8, 25.5, 22.5, "Downstream\nutility", "degeneration and\nnarrowing: within\ndomain and across\nvendors", C_EVAL)
box(93.5, 0.8, 25.5, 22.5, "Series\nnetwork", "k-NN graph over R3:\nassortativity,\ncommunities,\npatient-level null", C_EVAL)

fig.subplots_adjust(0, 0, 1, 1)
fig.savefig("csonet/figures/fig0_pipeline.pdf")
fig.savefig("csonet/figures/fig0_pipeline.png", dpi=300)
print("saved fig0_pipeline")
