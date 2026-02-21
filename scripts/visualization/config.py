from __future__ import annotations

import matplotlib as mpl
import seaborn as sns


# Approximate useful figure widths for MICCAI/LNCS papers.
MICCAI_SINGLE_COLUMN_FIGSIZE = (3.4, 2.4)
MICCAI_DOUBLE_COLUMN_FIGSIZE = (6.8, 2.4)

MICCAI_MPL_PARAMS = {
    "figure.dpi": 120,
    "figure.figsize": MICCAI_DOUBLE_COLUMN_FIGSIZE,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "figure.titlesize": 10,
    "figure.labelsize": 9,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_miccai_style() -> None:
    """Apply a publication-oriented style for MICCAI figures."""
    sns.set_theme(style="whitegrid", context="paper")
    mpl.rcParams.update(MICCAI_MPL_PARAMS)
