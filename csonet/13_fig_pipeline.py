"""CSoNet camera-ready: study-design overview figure (reviewer 2).
Drawn at the LNCS text width (12.2 cm = 4.8 in) so fonts print at their nominal size; no text below 6 pt (LNCS Sect. 4.5).
Data flow: SPIDER -> four representations -> ONE arrow into a group of four PARALLEL
analyses (no arrows between analyses, since none feeds another). The spine graph is a
detail of R4 and is linked to it by a dashed line, not by a data-flow arrow.

Robustness: fonts are embedded as TrueType (fonttype 42) so every PDF viewer renders
the same glyph widths, and the script refuses to save if any text block comes closer
than MARGIN (about 3 pt) to the border of the box that contains it.
No dash glyphs in any label."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse

plt.rcParams.update({"font.family": "serif", "font.size": 6.0,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
W, H = 4.8, 2.05
fig, ax = plt.subplots(figsize=(W, H))
fig.subplots_adjust(0, 0, 1, 1)
ax.set_xlim(0, 122); ax.set_ylim(0, 54); ax.axis("off")
C_IN, C_REP, C_EVAL, C_GRP = "#e8eef7", "#fdf1e0", "#e7f3ea", "#f7faf7"
MARGIN = 1.2          # minimum gap to the box border, data units (about 3 pt)
CHECKS = []          # (text artist, (x, y, w, h) container in data units)


def circle(xy, r, **kw):
    """Round node in data units despite unequal x/y scaling of the axes."""
    k = (54 / H) / (122 / W)
    return Ellipse(xy, 2 * r, 2 * r * k, **kw)


def box(x, y, w, h, title, body, fc, tfs=6.5, bfs=6.0, uses=None):
    """uses: which representations the analysis takes as input (italic footer line)."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.5",
                                fc=fc, ec="#555", lw=0.6))
    t = ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", fontsize=tfs,
                weight="bold", linespacing=1.05)
    title_h = 1.6 + 3.3 * (title.count("\n") + 1)       # title block height, data units
    foot_h = 1.5 + 2.6 if uses else 0.0
    b = ax.text(x + w / 2, y + foot_h + (h - title_h - foot_h) / 2, body, ha="center",
                va="center", fontsize=bfs, linespacing=1.2)
    CHECKS.extend([(t, (x, y, w, h)), (b, (x, y, w, h))])
    if uses:   # appended right after the body so the overlap check compares the two
        u = ax.text(x + w / 2, y + 1.5, uses, ha="center", va="bottom", fontsize=bfs,
                    style="italic", color="#1f4e79")
        CHECKS.append((u, (x, y, w, h)))


def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#333", mutation_scale=7,
                                shrinkA=0, shrinkB=0))


# ---------------------------------------------------------------- input
box(0.4, 8.5, 16.2, 38, "SPIDER",
    "447 series\n218 patients\n4 hospitals\n\nexpert\nmasks\n\nvendor,\nfield", C_IN)

# ---------------------------------------------------------------- representations
RH = 12.2            # height of a representation box, data units
reps = [(41.2, "R1, R2 intensity", "raw signal in masks;\nratios, z scores"),
        (28.2, "R3 mask geometry", "volumes, spacing,\ncounts, voxel size"),
        (15.2, "R4 graph-relational", "Laplacian spectrum\n+ scale-free ratios")]
for y, t, b in reps:
    box(19.5, y, 28.5, RH, t, b, C_REP, tfs=6.2)
    arrow(16.6, 27.5, 19.3, y + RH / 2)

# collector: all representations -> analyses (one data-flow arrow)
xb = 50.0
ymid = [y + RH / 2 for y, _, _ in reps]
for ym in ymid:
    ax.plot([48.0, xb], [ym, ym], c="#333", lw=0.7)
ax.plot([xb, xb], [min(ymid), max(ymid)], c="#333", lw=0.7)
arrow(xb, sum(ymid) / 3, 62.8, sum(ymid) / 3)

# ---------------------------------------------------------------- spine graph (detail of R4)
px, py, pw, ph = 17.0, 0.4, 37.0, 13.4
ax.add_patch(FancyBboxPatch((px, py), pw, ph, boxstyle="round,pad=0,rounding_size=1.2",
                            fc="white", ec="#999", lw=0.5, ls=(0, (3, 2))))
ax.plot([33.7, 33.7], [15.2, py + ph], c="#999", lw=0.5, ls=(0, (2, 1.5)))
xs = [px + 3.4 + 6.8 * i for i in range(3)]; yv = py + 9.4; cx, cy = xs[1], py + 3.3
for i, x in enumerate(xs):
    if i:
        ax.plot([xs[i - 1], x], [yv, yv], c="#444", lw=0.6, zorder=2)
        ax.add_patch(circle(((xs[i - 1] + x) / 2, yv), 1.0, fc="#e0a458", ec="k", lw=0.3, zorder=3))
    ax.plot([x, cx], [yv, cy], c="#9aa", lw=0.5, zorder=1)
    ax.add_patch(circle((x, yv), 2.2, fc="#5b7fb5", ec="k", lw=0.3, zorder=3))
    ax.text(x, yv, f"V{i + 1}", ha="center", va="center", fontsize=6.0, color="w", zorder=4)
ax.add_patch(circle((cx, cy), 2.2, fc="#6aa56e", ec="k", lw=0.3, zorder=3))
ax.text(cx, cy, "C", ha="center", va="center", fontsize=6.0, color="w", zorder=4)
leg = ax.text(px + pw - 1.5, py + ph / 2, "V vertebra\nD disc\nC canal", ha="right",
              va="center", fontsize=6.0, linespacing=1.15)
CHECKS.append((leg, (px, py, pw, ph)))

# ---------------------------------------------------------------- parallel analyses
gx, gy, gw, gh = 63.0, 0.4, 58.6, 53.2
ax.add_patch(FancyBboxPatch((gx, gy), gw, gh, boxstyle="round,pad=0,rounding_size=1.8",
                            fc=C_GRP, ec="#888", lw=0.6))
gt = ax.text(gx + gw / 2, gy + gh - 1.2, "Parallel analyses (same protocol)", ha="center",
             va="top", fontsize=6.5, weight="bold")
CHECKS.append((gt, (gx, gy, gw, gh)))
bw, bh = 27.0, 22.8
# input of each analysis, as in the paper: Table 1 (R1 to R4), Table 2 and ComBat (R3, R4),
# Table 3 (R3, R4), Sec. 5.5 (R3)
box(64.8, 25.8, bw, bh, "Scanner\npredictability", "random forest,\ngrouped CV,\nbootstrap,\npermutation", C_EVAL, uses="on R1 to R4")
box(93.4, 25.8, bw, bh, "Acquisition\ncontrols", "voxel size,\ncoverage, sequence,\nfixed field,\ncase mix, ComBat", C_EVAL, uses="on R3, R4")
box(64.8, 1.6, bw, bh, "Downstream\nutility", "degeneration,\nnarrowing;\nwithin and\nacross vendors", C_EVAL, uses="on R3, R4")
box(93.4, 1.6, bw, bh, "Series\nnetwork", "k-NN graph,\nassortativity,\ncommunities,\npatient null", C_EVAL, uses="on R3")

# ---------------------------------------------------------------- containment check
fig.canvas.draw()
r = fig.canvas.get_renderer(); inv = ax.transData.inverted()
bad = []
for t, (x, y, w, h) in CHECKS:
    bb = t.get_window_extent(renderer=r)
    (x0, y0), (x1, y1) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
    mx = my = MARGIN
    if x0 < x + mx or x1 > x + w - mx or y0 < y + my or y1 > y + h - my:
        bad.append((t.get_text().replace("\n", " | ")[:40],
                    round(x0 - x, 2), round(x + w - x1, 2), round(y0 - y, 2), round(y + h - y1, 2)))
# title and body inside the same box must not overlap each other
for i in range(0, len(CHECKS) - 1):
    t1, c1 = CHECKS[i]; t2, c2 = CHECKS[i + 1]
    if c1 == c2:
        b1 = t1.get_window_extent(renderer=r); b2 = t2.get_window_extent(renderer=r)
        if b1.overlaps(b2):
            bad.append(("overlap: " + t1.get_text()[:20], 0, 0, 0, 0))
if bad:
    for b in bad:
        print("TOO TIGHT (left,right,bottom,top gaps):", b)
    raise SystemExit("figure NOT saved: text too close to a border")

fig.savefig("csonet/figures/fig0_pipeline.pdf")
fig.savefig("csonet/figures/fig0_pipeline.png", dpi=300)
print("saved fig0_pipeline; all", len(CHECKS), "text blocks inside their boxes with",
      MARGIN, "data-unit margin (about 3 pt)")
