import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

OUT = r"C:\Users\naima\DySCo-main\DySCo-main\DELFT_NEW\dissertation_figures_final"
os.makedirs(OUT, exist_ok=True)

NAVY   = "#1a2e4a"
WHITE  = "#ffffff"
STRIPE = "#f4f6f9"
BORDER = "#aaaaaa"


def draw_table(ax, col_headers, rows, col_widths=None):
    n_cols = len(col_headers)
    n_rows = len(rows)
    total  = sum(col_widths) if col_widths else n_cols
    if col_widths is None:
        col_widths = [1.0] * n_cols

    row_h = 1.0 / (n_rows + 1)
    x_starts = []
    x = 0
    for w in col_widths:
        x_starts.append(x / total)
        x += w

    def cell(ax, x0, y0, w, h, text, bg, fg, bold=False, align="center"):
        rect = mpatches.FancyBboxPatch(
            (x0, y0), w / total, h,
            boxstyle="square,pad=0",
            linewidth=0.6, edgecolor=BORDER,
            facecolor=bg, transform=ax.transAxes, clip_on=False
        )
        ax.add_patch(rect)
        ha  = "left" if align == "left" else "center"
        xpos = x0 + (0.012 if align == "left" else (w / total) / 2)
        ax.text(xpos, y0 + h / 2, text,
                transform=ax.transAxes,
                ha=ha, va="center",
                fontsize=8.5, color=fg,
                fontweight="bold" if bold else "normal",
                clip_on=False)

    # header
    y_top = 1.0 - row_h
    for i, (hdr, w) in enumerate(zip(col_headers, col_widths)):
        cell(ax, x_starts[i], y_top, w, row_h, hdr, NAVY, WHITE,
             bold=True, align="left" if i == 0 else "center")

    # data rows
    for r, row in enumerate(rows):
        y  = y_top - (r + 1) * row_h
        bg = WHITE if r % 2 == 0 else STRIPE
        is_last = (r == len(rows) - 1)
        for i, (val, w) in enumerate(zip(row, col_widths)):
            cell(ax, x_starts[i], y, w, row_h, str(val),
                 bg if not is_last else "#e8ecf1",
                 NAVY, bold=is_last,
                 align="left" if i == 0 else "center")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


# ── TABLE 2 ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 2.3))
fig.patch.set_facecolor("white")
ax.set_position([0.02, 0.22, 0.96, 0.72])

draw_table(ax,
    col_headers=["Metric", "Cartoon Mean", "Rest Mean", "W", "p", "r"],
    rows=[
        ["Von Neumann Entropy",    "2.177",  "2.145",  "2.0", "0.133", "0.188"],
        ["Reconfiguration Speed",  "36,949", "37,183", "5.0", "0.333", "0.625"],
        ["Connectivity Norm (L2)", "28,377", "30,043", "2.0", "0.133", "0.188"],
    ],
    col_widths=[3.2, 1.5, 1.5, 0.8, 0.8, 0.8]
)
fig.text(0.03, 0.14,
    "Table 2. Group-level Wilcoxon signed-rank test results comparing cartoon and resting-state conditions across five paediatric participants.\n"
    "W = test statistic. p = two-sided p-value. r = rank-biserial correlation effect size. Minimum achievable p with n = 5 is 0.0625.\n"
    "No significant differences observed.",
    fontsize=7.5, color="#444444", va="top", style="italic")

fig.savefig(os.path.join(OUT, "table2_wilcoxon.png"), dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Saved table2_wilcoxon.png")


# ── TABLE 3 ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.5, 2.6))
fig.patch.set_facecolor("white")
ax.set_position([0.02, 0.20, 0.96, 0.74])

draw_table(ax,
    col_headers=["Participant", "Entropy", "Speed", "Norm (L2)"],
    rows=[
        ["P001", "Cartoon > Rest", "Rest > Cartoon", "Rest > Cartoon"],
        ["P002", "Cartoon > Rest", "Rest > Cartoon", "Rest > Cartoon"],
        ["P003", "Cartoon > Rest", "Rest > Cartoon", "Rest > Cartoon"],
        ["P004", "Cartoon > Rest", "Rest > Cartoon", "Rest > Cartoon"],
        ["P005", "Rest > Cartoon", "Cartoon > Rest", "Cartoon > Rest"],
    ],
    col_widths=[1.5, 2.5, 2.5, 2.5]
)
fig.text(0.03, 0.11,
    "Table 3. Direction of condition effects for each paediatric participant. Four of five participants showed cartoon > rest for entropy\n"
    "and rest > cartoon for speed and norm. P005 showed the reversed direction across all three metrics.",
    fontsize=7.5, color="#444444", va="top", style="italic")

fig.savefig(os.path.join(OUT, "table3_direction.png"), dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Saved table3_direction.png")


# ── TABLE 4 ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 2.9))
fig.patch.set_facecolor("white")
ax.set_position([0.02, 0.18, 0.96, 0.76])

draw_table(ax,
    col_headers=["Participant", "Cartoon\nEntropy", "Rest\nEntropy",
                 "Cartoon\nSpeed", "Rest\nSpeed", "Cartoon\nNorm", "Rest\nNorm"],
    rows=[
        ["P001",       "2.157", "2.076", "35,212", "35,539", "28,696", "32,550"],
        ["P002",       "2.249", "2.200", "32,985", "33,487", "24,554", "26,973"],
        ["P003",       "2.141", "2.098", "41,026", "42,657", "30,577", "33,763"],
        ["P004",       "2.134", "2.116", "39,943", "41,891", "31,464", "32,510"],
        ["P005",       "2.205", "2.236", "35,578", "32,339", "26,596", "24,418"],
        ["Group Mean", "2.177", "2.145", "36,949", "37,183", "28,377", "30,043"],
    ],
    col_widths=[1.8, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]
)
fig.text(0.03, 0.09,
    "Table 4. Subject-level condition means for three DySCo metrics across the paediatric cohort (n = 5). Group means shown in final row.\n"
    "These values formed the input to the Wilcoxon signed-rank comparisons in Table 2.",
    fontsize=7.5, color="#444444", va="top", style="italic")

fig.savefig(os.path.join(OUT, "table4_subject_means.png"), dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Saved table4_subject_means.png")
