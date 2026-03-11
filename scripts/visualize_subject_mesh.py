#!/usr/bin/env python
"""CLI helper for visualising a single subject's cortical surface mesh graph.

Usage
-----
Plotly interactive (loads from Parquet):

.. code-block:: bash

    python visualize_subject_mesh.py \\
        --subject 100307 \\
        --graph_dir /path/to/results/default/graphs \\
        --method plotly

    # Save to HTML instead of opening a browser window
    python visualize_subject_mesh.py \\
        --subject 100307 \\
        --graph_dir /path/to/results/default/graphs \\
        --method plotly \\
        --output_html /tmp/100307_mesh.html

nilearn static (loads from Parquet and reconstructs SurfaceMeshData):

.. code-block:: bash

    python visualize_subject_mesh.py \\
        --subject 100307 \\
        --graph_dir /path/to/results/default/graphs \\
        --method nilearn \\
        --mode parcel \\
        --hemi left

Common options
--------------
--show_edges     Include mesh edges in the Plotly plot (slow for large meshes).
--feature_index  Which feature column to colour by (default: 0).
--output_html    Save Plotly figure as standalone HTML.
--output_png     Save nilearn figure as PNG.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Visualise a cortical surface mesh for a single subject.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required
    p.add_argument(
        "--subject",
        required=True,
        metavar="SUBJECT_ID",
        help="Subject identifier, e.g. '100307'.",
    )
    p.add_argument(
        "--graph_dir",
        required=True,
        type=Path,
        metavar="DIR",
        help=(
            "Directory containing {subject_id}_nodes.parquet and "
            "{subject_id}_edges.parquet files."
        ),
    )
    p.add_argument(
        "--method",
        required=True,
        choices=["plotly", "nilearn"],
        help="Visualisation backend.",
    )

    # Shared optional
    p.add_argument(
        "--feature_index",
        type=int,
        default=0,
        metavar="N",
        help="Which feature column to colour by (0-based, default: 0).",
    )

    # Plotly-specific
    p.add_argument(
        "--show_edges",
        action="store_true",
        default=False,
        help=(
            "[Plotly] Render mesh edges as grey lines. "
            "Disabled by default — can be slow for ~60k-vertex meshes."
        ),
    )
    p.add_argument(
        "--output_html",
        type=Path,
        default=None,
        metavar="FILE",
        help="[Plotly] Save figure as standalone HTML file.",
    )

    # nilearn-specific
    p.add_argument(
        "--mode",
        choices=["feature", "parcel"],
        default="feature",
        help="[nilearn] Colour surface by 'feature' values or 'parcel' labels.",
    )
    p.add_argument(
        "--hemi",
        choices=["left", "right"],
        default="left",
        help="[nilearn] Which hemisphere to display (default: left).",
    )
    p.add_argument(
        "--view",
        default="lateral",
        help="[nilearn] Viewing angle (default: 'lateral').",
    )
    p.add_argument(
        "--output_png",
        type=Path,
        default=None,
        metavar="FILE",
        help="[nilearn] Save figure as PNG file.",
    )

    # Atlas / filename identity
    p.add_argument(
        "--metric",
        default="md",
        metavar="METRIC",
        help="Microstructure metric used in the filename (e.g. 'md', 'ndi'). Default: md",
    )
    p.add_argument(
        "--tissue_type",
        default="gray",
        metavar="TISSUE",
        help="Tissue type used in the filename (e.g. 'gray', 'white'). Default: gray",
    )
    p.add_argument(
        "--atlas_name",
        default="schaefer",
        metavar="ATLAS",
        help="Atlas name used in the filename (e.g. 'schaefer'). Default: schaefer",
    )
    p.add_argument(
        "--n_parcels",
        type=int,
        default=1000,
        metavar="N",
        help="Number of parcels used in the filename (e.g. 1000). Default: 1000",
    )

    return p


def _run_plotly(args: argparse.Namespace) -> None:
    from diff_benchmark.preprocessing.utils.utils_mesh_visualization import (
        plot_mesh_plotly,
    )

    fig = plot_mesh_plotly(
        subject_id=args.subject,
        graph_dir=args.graph_dir,
        show_edges=args.show_edges,
        feature_index=args.feature_index,
        output_html=args.output_html,
    )
    if args.output_html is None:
        fig.show()
    else:
        print(f"Saved interactive HTML → {args.output_html}")


def _run_nilearn(args: argparse.Namespace) -> None:
    from diff_benchmark.preprocessing.utils.utils_graph_export import (
        load_graph_from_parquet,
    )
    from diff_benchmark.preprocessing.utils.utils_mesh_visualization import (
        plot_mesh_nilearn,
    )

    # Reconstruct SurfaceMeshData from Parquet (dataset-agnostic)
    mesh = load_graph_from_parquet(
        subject_id=args.subject,
        graph_dir=args.graph_dir,
        metric=args.metric,
        tissue_type=args.tissue_type,
        atlas_name=args.atlas_name,
        n_parcels=args.n_parcels,
    )

    output_file = args.output_png

    display = plot_mesh_nilearn(
        mesh=mesh,
        mode=args.mode,
        feature_index=args.feature_index,
        hemi=args.hemi,
        view=args.view,
        output_file=output_file,
    )

    if output_file is not None:
        print(f"Saved figure → {output_file}")
    else:
        try:
            display.show()
        except AttributeError:
            # nilearn display objects behave differently depending on backend;
            # the figure is already shown inline in Jupyter.
            pass


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.graph_dir.exists():
        print(
            f"ERROR: graph_dir does not exist: {args.graph_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.method == "plotly":
        _run_plotly(args)
    elif args.method == "nilearn":
        _run_nilearn(args)


if __name__ == "__main__":
    main()
