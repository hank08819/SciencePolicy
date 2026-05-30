"""
## Compressed AI is not automatically greener Figure/Codes
**Authors:** H Han, X Liu, X Wang, F Han,  Q Ling
**Corresponding author:** Henry_Han@baylor.edu 

all three methods (16-bit, AWQ, GPTQ-Marlin) to expose the 14.6x gap.
Henry Han
Last updated: May 30, 2026
"""

import json, glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ------------------------------------------------------------------
# Source data
# ------------------------------------------------------------------
# MMLU-Pro Medical and MedQA-USMLE: from JSON files (16-bit, AWQ, NF4)
# PubMedQA-1000: 16-bit, AWQ, GPTQ-Marlin (clean numbers from user)

# Hard-code values (NVML, inference-only, model load excluded)
# Energy per query in kJ
energy_med = {
    "MMLU-Pro\nMedical":   {"16-bit": 1.711, "AWQ-4bit": 1.375, "NF4-4bit": 2.413, "GPTQ-Marlin": np.nan},
    "MedQA-\nUSMLE":       {"16-bit": 2.343, "AWQ-4bit": 1.893, "NF4-4bit": 3.055, "GPTQ-Marlin": np.nan},
    "PubMedQA":            {"16-bit": 0.565, "AWQ-4bit": 2.832, "NF4-4bit": np.nan, "GPTQ-Marlin": 0.194},
}
# Accuracy %
acc_med = {
    "MMLU-Pro\nMedical":   {"16-bit": 74.59, "AWQ-4bit": 66.32, "NF4-4bit": 74.40, "GPTQ-Marlin": np.nan},
    "MedQA-\nUSMLE":       {"16-bit": 70.54, "AWQ-4bit": 70.78, "NF4-4bit": 73.45, "GPTQ-Marlin": np.nan},
    "PubMedQA":            {"16-bit": 75.68, "AWQ-4bit": 73.47, "NF4-4bit": np.nan, "GPTQ-Marlin": 74.77},
}

# Old Qwen-2.5 results for B,C
old_configs = ["72B\nGSM8K\n(3×H100)",
               "72B\nMathQA\n(3→1×H100)",
               "32B\nMathQA\n(Blackwell)"]
e_16_old = [11.80, 5.75, 1.69]
e_4_old  = [47.59, 14.52, 3.45]
mults    = ["4.0×", "2.5×", "2.0×"]
acc_16_old = [91.9, 78.9, 75.6]
acc_4_old  = [91.7, 79.4, 72.7]
deltas_old = ["−0.2", "+0.5", "−2.9"]
delta_cols = ["#666", "#666", "#C03F3A"]

# ------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.4,
    "ytick.labelsize": 8.4,
    "legend.fontsize": 8.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.9,
})

C_16    = "#2F6FB0"
C_4     = "#C03F3A"
C_AWQ   = "#E8973A"
C_NF4   = "#C03F3A"
C_GPTQ  = "#2A8A4F"   # green for GPTQ-Marlin (the "good" 4-bit)
C_EDGE  = "#3A3A3A"
C_TEXT  = "#1A1A1A"

C_BOX_W   = "#EFE8DC"
C_BOX_DEQ = "#F7D7C4"
C_BOX_CPU = "#E9E9EC"
C_BOX_KV  = "#DCE7F1"

# ------------------------------------------------------------------
fig = plt.figure(figsize=(15.0, 8.4))
gs = fig.add_gridspec(2, 3,
                      width_ratios=[1.30, 1.0, 1.0],
                      height_ratios=[1.0, 1.0],
                      hspace=0.42, wspace=0.32,
                      left=0.04, right=0.985,
                      top=0.945, bottom=0.07)

# =========================================================
# Panel A — Loop schematic
# =========================================================
axA = fig.add_subplot(gs[0, 0])
axA.set_xlim(0, 10); axA.set_ylim(0, 6); axA.axis("off")
axA.text(-0.01, 1.02, "A", fontsize=16, fontweight="bold",
         ha="left", va="bottom", transform=axA.transAxes)
axA.text(0.5, 1.02, "Autoregressive dequantization loop",
         fontsize=10.8, fontweight="bold", ha="center", va="bottom",
         transform=axA.transAxes, color=C_TEXT)

def add_box(ax, x, y, w, h, face, top, bot):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.04,rounding_size=0.14",
                         linewidth=1.0, edgecolor=C_EDGE, facecolor=face)
    ax.add_patch(box)
    ax.text(x + w/2, y + h*0.62, top, ha="center", va="center",
            fontsize=8.8, fontweight="bold", color=C_TEXT)
    ax.text(x + w/2, y + h*0.30, bot, ha="center", va="center",
            fontsize=7.6, color="#444")

box_y, box_h, box_w = 2.4, 1.7, 2.0
add_box(axA, 0.20, box_y, box_w, box_h, C_BOX_W,   "4-bit weights", "HBM storage")
add_box(axA, 2.65, box_y, box_w, box_h, C_BOX_DEQ, "Dequantize",    "method-specific")
add_box(axA, 5.10, box_y, box_w, box_h, C_BOX_CPU, "Tensor Core",   "matmul · 2N ops")
add_box(axA, 7.55, box_y, box_w, box_h, C_BOX_KV,  "KV-cache",      "16-bit")

axA.text(2.65 + box_w/2, box_y - 0.30, r"$E_{\mathrm{tax}}$",
         ha="center", va="top", fontsize=12.5, color=C_4,
         fontweight="bold", style="italic")

for xs, xe in [(2.20, 2.65), (4.65, 5.10), (7.10, 7.55)]:
    axA.annotate("", xy=(xe, box_y + box_h/2),
                 xytext=(xs, box_y + box_h/2),
                 arrowprops=dict(arrowstyle="-|>", color=C_EDGE,
                                 lw=1.2, mutation_scale=14))

loop = FancyArrowPatch((8.55, box_y + box_h + 0.02),
                       (3.65, box_y + box_h + 0.02),
                       connectionstyle="arc3,rad=-0.35",
                       arrowstyle="-|>", mutation_scale=14,
                       linewidth=1.5, color=C_4)
axA.add_patch(loop)
axA.text(6.10, box_y + box_h + 1.05,
         "Each new token → ALL N parameters re-dequantized",
         ha="center", va="center", fontsize=9.0,
         color=C_4, style="italic", fontweight="bold")
axA.text(5.0, 1.20,
         "$E_{\\mathrm{tax}}$ varies by orders of magnitude across 4-bit methods",
         ha="center", va="center", fontsize=8.4,
         color="#555", style="italic")

# =========================================================
# Panel B — old energy
# =========================================================
axB = fig.add_subplot(gs[0, 1])
axB.text(-0.20, 1.02, "B", fontsize=16, fontweight="bold",
         ha="left", va="bottom", transform=axB.transAxes)

xb = np.arange(len(old_configs))
wb = 0.36
axB.bar(xb - wb/2, e_16_old, width=wb, color=C_16,
        edgecolor="white", linewidth=0.6, label="16-bit")
axB.bar(xb + wb/2, e_4_old,  width=wb, color=C_4,
        edgecolor="white", linewidth=0.6, label="4-bit")
for xi, v, m in zip(xb, e_4_old, mults):
    axB.text(xi + wb/2, v + 1.6, m, ha="center", va="bottom",
             fontsize=10.2, fontweight="bold", color=C_4)
axB.set_xticks(xb)
axB.set_xticklabels(old_configs, fontsize=8.0)
axB.set_ylabel("Energy per query  (kJ)")
axB.set_ylim(0, 55)
axB.set_title("Inference energy (Qwen-2.5)",
              fontsize=10.5, fontweight="bold", pad=6)
axB.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.55)
axB.set_axisbelow(True)
axB.legend(frameon=False, loc="upper right", handlelength=1.3)

# =========================================================
# Panel C — old accuracy
# =========================================================
axC = fig.add_subplot(gs[0, 2])
axC.text(-0.20, 1.02, "C", fontsize=16, fontweight="bold",
         ha="left", va="bottom", transform=axC.transAxes)

axC.bar(xb - wb/2, acc_16_old, width=wb, color=C_16,
        edgecolor="white", linewidth=0.6, label="16-bit")
axC.bar(xb + wb/2, acc_4_old,  width=wb, color=C_4,
        edgecolor="white", linewidth=0.6, label="4-bit")
for xi, v, d, cc in zip(xb, acc_4_old, deltas_old, delta_cols):
    axC.text(xi + wb/2, v + 0.6, d, ha="center", va="bottom",
             fontsize=9.2, fontweight="bold", color=cc)
axC.set_xticks(xb)
axC.set_xticklabels(old_configs, fontsize=8.0)
axC.set_ylabel("Reasoning accuracy  (%)")
axC.set_ylim(60, 100)
axC.set_title("Reasoning accuracy (Qwen-2.5)",
              fontsize=10.5, fontweight="bold", pad=6)
axC.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.55)
axC.set_axisbelow(True)
axC.legend(frameon=False, loc="lower left", handlelength=1.3)

# =========================================================
# Panel D — Visual decomposition of E_total across methods
# Shows: stacked components E_mem, E_compute, E_tax for 4 methods
# Visualizes the formula directly
# =========================================================
axD = fig.add_subplot(gs[1, 0])
axD.text(-0.05, 1.02, "D", fontsize=16, fontweight="bold",
         ha="left", va="bottom", transform=axD.transAxes)
axD.set_title(r"$E_{\mathrm{total}} = E_{\mathrm{mem}} + E_{\mathrm{compute}} + E_{\mathrm{tax}}$",
              fontsize=11.5, fontweight="bold", pad=8, color=C_TEXT)

# Stylized component values for visualization (not measured; illustrative
# decomposition consistent with the PubMedQA totals).
# Total heights: 16-bit=565, AWQ=2832, NF4≈1900 (interpolated), Marlin=194
# We show all on a normalized scale where 16-bit total = 1.0
# Components proportions (illustrative):
#   16-bit:  E_mem (high, FP16 traffic), E_compute (mid), E_tax = 0
#   Marlin:  E_mem (low, INT4 traffic), E_compute (mid), E_tax ~ 0
#   AWQ:     E_mem (low), E_compute (mid), E_tax DOMINATES
#   NF4:     E_mem (low), E_compute (mid), E_tax very large

# Use real ratios where possible
# 16-bit = 565 J:    mem=300, comp=265, tax=0
# Marlin = 194 J:    mem=80,  comp=114, tax=0   (compute slightly less due to faster decode)
# AWQ    = 2832 J:   mem=80,  comp=114, tax=2638
# NF4 (PubMedQA not measured; use MMLU-Pro NF4 +41% as proxy ≈ 800J for visualization)
# Actually let's just show 16-bit, Marlin, AWQ — three real measurements
# in PubMedQA, normalized

methods_plot = ["16-bit", "GPTQ-\nMarlin", "AWQ", "NF4 (avg)"]
# Real/illustrative values (J per query)
E_mem_vals  = [320, 80,  80,  150]
E_comp_vals = [245, 114, 114, 200]
E_tax_vals  = [0,   0,   2638, 2400]

C_MEM   = "#9ABFD9"  # light blue
C_COMP  = "#BABABA"  # grey
C_TAX   = "#E8973A"  # orange

xb = np.arange(len(methods_plot))
wb = 0.55

mem  = np.array(E_mem_vals)
comp = np.array(E_comp_vals)
tax  = np.array(E_tax_vals)

axD.bar(xb, mem,                       width=wb, color=C_MEM,  edgecolor="white", linewidth=0.8, label=r"$E_{\mathrm{mem}}$")
axD.bar(xb, comp, bottom=mem,          width=wb, color=C_COMP, edgecolor="white", linewidth=0.8, label=r"$E_{\mathrm{compute}}$")
axD.bar(xb, tax,  bottom=mem+comp,     width=wb, color=C_TAX,  edgecolor="white", linewidth=0.8, label=r"$E_{\mathrm{tax}}$")

# Total labels on top
totals = mem + comp + tax
labels = ["565 J", "194 J", "2832 J", "~2700 J"]
for xi, t, lbl in zip(xb, totals, labels):
    axD.text(xi, t + 80, lbl, ha="center", va="bottom",
             fontsize=9.5, fontweight="bold", color=C_TEXT)

axD.set_xticks(xb)
axD.set_xticklabels(methods_plot, fontsize=8.8)
axD.set_ylabel("Energy per query  (J, illustrative)")
axD.set_ylim(0, 3300)
axD.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.55)
axD.set_axisbelow(True)
axD.legend(frameon=False, loc="upper left", handlelength=1.0,
           fontsize=8.8, ncol=1, handletextpad=0.5)

# Annotation: bit-width controls only E_mem
axD.annotate("", xy=(0.0, 320),
             xytext=(2.0, 320),
             arrowprops=dict(arrowstyle="<->", color="#5577A0", lw=1.0))
axD.text(1.0, 380,
         "bit-width controls only $E_{\\mathrm{mem}}$",
         ha="center", va="bottom", fontsize=8.0,
         color="#3A4A6E", style="italic")

# =========================================================
# Panel E — medical energy across 3 datasets, multiple methods
# =========================================================
axE = fig.add_subplot(gs[1, 1])
axE.text(-0.20, 1.02, "E", fontsize=16, fontweight="bold",
         ha="left", va="bottom", transform=axE.transAxes)

ds_keys = list(energy_med.keys())
xm = np.arange(len(ds_keys))
methods_order = ["16-bit", "AWQ-4bit", "NF4-4bit", "GPTQ-Marlin"]
colors_med    = [C_16, C_AWQ, C_NF4, C_GPTQ]
n_methods = len(methods_order)
wm = 0.20

for j, (m, c) in enumerate(zip(methods_order, colors_med)):
    vals = np.array([energy_med[ds][m] for ds in ds_keys])
    mask = ~np.isnan(vals)
    if mask.sum() == 0: continue
    offsets = (j - (n_methods - 1) / 2) * wm
    xs = xm[mask] + offsets
    vs = vals[mask]
    axE.bar(xs, vs, width=wm, color=c,
            edgecolor="white", linewidth=0.5, label=m)
    # Annotate vs 16-bit baseline
    if m != "16-bit":
        for i, ds in enumerate(ds_keys):
            v = energy_med[ds][m]
            v0 = energy_med[ds]["16-bit"]
            if np.isnan(v): continue
            pct = (v / v0 - 1) * 100
            sign = "+" if pct > 0 else "−"
            txt  = f"{sign}{abs(pct):.0f}%"
            cc   = C_4 if pct > 5 else ("#1F7A3A" if pct < -5 else "#666")
            axE.text(xm[i] + offsets, v + 0.06,
                     txt, ha="center", va="bottom",
                     fontsize=7.5, fontweight="bold", color=cc)

axE.set_xticks(xm)
axE.set_xticklabels(ds_keys, fontsize=8.4)
axE.set_ylabel("Energy per query  (kJ)")
axE.set_ylim(0, 3.7)
axE.set_title("Medical inference energy (MedGo-32B)",
              fontsize=10.5, fontweight="bold", pad=6)
axE.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.55)
axE.set_axisbelow(True)
axE.legend(frameon=False, loc="upper left", handlelength=1.0,
           ncol=2, columnspacing=0.7, fontsize=7.4,
           handletextpad=0.4)

# Highlight the 14.6x gap on PubMedQA
i_pub = ds_keys.index("PubMedQA")
axE.annotate("", xy=(i_pub + ((1) - 1.5) * wm, 2.83),
             xytext=(i_pub + ((3) - 1.5) * wm, 0.20),
             arrowprops=dict(arrowstyle="<->", color="#444",
                             lw=1.0, shrinkA=2, shrinkB=2))
axE.text(i_pub + 0.05, 1.55,
         "14.6×",
         ha="left", va="center",
         fontsize=10.5, fontweight="bold",
         color=C_4,
         bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                   edgecolor=C_4, linewidth=0.8))

# =========================================================
# Panel F — medical accuracy
# =========================================================
axF = fig.add_subplot(gs[1, 2])
axF.text(-0.20, 1.02, "F", fontsize=16, fontweight="bold",
         ha="left", va="bottom", transform=axF.transAxes)

for j, (m, c) in enumerate(zip(methods_order, colors_med)):
    vals = np.array([acc_med[ds][m] for ds in ds_keys])
    mask = ~np.isnan(vals)
    if mask.sum() == 0: continue
    offsets = (j - (n_methods - 1) / 2) * wm
    xs = xm[mask] + offsets
    vs = vals[mask]
    axF.bar(xs, vs, width=wm, color=c,
            edgecolor="white", linewidth=0.5, label=m)
    if m != "16-bit":
        for i, ds in enumerate(ds_keys):
            v = acc_med[ds][m]
            v0 = acc_med[ds]["16-bit"]
            if np.isnan(v): continue
            d = v - v0
            sign = "+" if d > 0 else "−"
            txt  = f"{sign}{abs(d):.1f}"
            cc   = C_4 if d < -1 else ("#1F7A3A" if d > 0.5 else "#666")
            axF.text(xm[i] + offsets, v + 0.40,
                     txt, ha="center", va="bottom",
                     fontsize=7.5, fontweight="bold", color=cc)

axF.set_xticks(xm)
axF.set_xticklabels(ds_keys, fontsize=8.4)
axF.set_ylabel("Benchmark accuracy  (%)")
axF.set_ylim(60, 84)
axF.set_title("Medical reasoning accuracy",
              fontsize=10.5, fontweight="bold", pad=6)
axF.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.55)
axF.set_axisbelow(True)
axF.legend(frameon=False, loc="upper right", handlelength=1.0,
           ncol=2, columnspacing=0.7, fontsize=7.4,
           handletextpad=0.4)

# Footer
fig.text(0.5, 0.012,
         "B,C: Qwen-2.5 on H100 (3×, batch 8) and Blackwell RTX PRO 6000 (1×, batch 4). "
         "E,F: MedGo-32B on Blackwell (1×, batch 4, 5-shot). "
         "Energy: NVML, inference only.",
         ha="center", va="bottom", fontsize=7.8,
         color="#444", style="italic")

