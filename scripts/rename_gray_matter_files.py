"""
Rename existing gray matter diffusion metric files to include tissue type.

Usage:
    python rename_gray_matter_files.py --base-dir /path/to/benchmark --datasets hcp camcan --dry-run
    python rename_gray_matter_files.py --base-dir /path/to/benchmark --datasets hcp camcan  # Actually rename
"""

import argparse
import shutil
from pathlib import Path
from typing import List


def rename_files(base_dir: Path, datasets: List[str], dry_run: bool = True):
    """
    Rename gray matter files to include tissue type.

    Args:
        base_dir: Base benchmark directory
        datasets: List of dataset names to process
        dry_run: If True, only print what would be renamed without actually renaming
    """
    total_renamed = 0
    total_errors = 0

    for dataset in datasets:
        dataset_dir = base_dir / dataset / "default" / "derivatives"

        if not dataset_dir.exists():
            print(f"⚠️  Dataset directory not found: {dataset_dir}")
            continue

        print(f"\n{'='*80}")
        print(f"Processing dataset: {dataset}")
        print(f"{'='*80}")

        # Find all subject directories
        subject_dirs = sorted(
            [
                d
                for d in dataset_dir.iterdir()
                if d.is_dir() and d.name.startswith("sub-")
            ]
        )
        print(f"Found {len(subject_dirs)} subjects\n")

        for subject_dir in subject_dirs:
            subject_id = subject_dir.name
            dwi_dir = subject_dir / "dwi"

            if not dwi_dir.exists():
                continue

            renamed_count = 0

            # Pattern 1: Hemisphere scalar files
            # From: sub-{id}_hemi-{L/R}_param-{metric}.scalar.gii
            # To:   sub-{id}_hemi-{L/R}_param-{metric}_tissue-gray.scalar.gii
            for scalar_file in dwi_dir.glob("sub-*_hemi-*_param-*.scalar.gii"):
                # Skip if already has tissue type
                if "_tissue-" in scalar_file.name:
                    continue

                # Parse filename
                parts = scalar_file.stem.replace(".scalar", "").split("_")

                # Reconstruct with tissue type before .scalar.gii
                new_name = scalar_file.name.replace(
                    ".scalar.gii", "_tissue-gray.scalar.gii"
                )
                new_path = scalar_file.parent / new_name

                if dry_run:
                    print(f"  [DRY RUN] {scalar_file.name}")
                    print(f"         -> {new_name}")
                else:
                    try:
                        scalar_file.rename(new_path)
                        print(f"  ✓ Renamed: {new_name}")
                        renamed_count += 1
                        total_renamed += 1
                    except Exception as e:
                        print(f"  ✗ Error renaming {scalar_file.name}: {e}")
                        total_errors += 1

            # Pattern 2: DWI map files
            # From: sub-{id}_param-{metric}_dwimap.nii.gz
            # To:   sub-{id}_param-{metric}_tissue-gray_dwimap.nii.gz
            for dwimap_file in dwi_dir.glob("sub-*_param-*_dwimap.nii.gz"):
                # Skip if already has tissue type
                if "_tissue-" in dwimap_file.name:
                    continue

                # Insert tissue type before _dwimap.nii.gz
                new_name = dwimap_file.name.replace(
                    "_dwimap.nii.gz", "_tissue-gray_dwimap.nii.gz"
                )
                new_path = dwimap_file.parent / new_name

                if dry_run:
                    print(f"  [DRY RUN] {dwimap_file.name}")
                    print(f"         -> {new_name}")
                else:
                    try:
                        dwimap_file.rename(new_path)
                        print(f"  ✓ Renamed: {new_name}")
                        renamed_count += 1
                        total_renamed += 1
                    except Exception as e:
                        print(f"  ✗ Error renaming {dwimap_file.name}: {e}")
                        total_errors += 1

            if renamed_count > 0 or dry_run:
                if dry_run:
                    files_to_rename = len(
                        list(dwi_dir.glob("sub-*_hemi-*_param-*.scalar.gii"))
                    ) + len(list(dwi_dir.glob("sub-*_param-*_dwimap.nii.gz")))
                    files_to_rename = sum(
                        1
                        for f in dwi_dir.glob("sub-*_hemi-*_param-*.scalar.gii")
                        if "_tissue-" not in f.name
                    )
                    files_to_rename += sum(
                        1
                        for f in dwi_dir.glob("sub-*_param-*_dwimap.nii.gz")
                        if "_tissue-" not in f.name
                    )
                    if files_to_rename > 0:
                        print(f"  [{subject_id}] Would rename {files_to_rename} files")

    print(f"\n{'='*80}")
    print(f"Summary:")
    print(f"{'='*80}")
    if dry_run:
        print(f"DRY RUN MODE - No files were actually renamed")
        print(f"Run without --dry-run to perform actual renaming")
    else:
        print(f"✓ Successfully renamed: {total_renamed} files")
        if total_errors > 0:
            print(f"✗ Errors: {total_errors} files")


def main():
    parser = argparse.ArgumentParser(
        description="Rename gray matter diffusion files to include tissue type"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        required=True,
        help="Base benchmark directory (e.g., /data/parietal/store3/work/ggomezji/benchmark)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="List of dataset names (e.g., hcp camcan wand)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without actually renaming files",
    )

    args = parser.parse_args()

    if not args.base_dir.exists():
        print(f"Error: Base directory does not exist: {args.base_dir}")
        return

    print(f"Base directory: {args.base_dir}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(
        f"Mode: {'DRY RUN (preview only)' if args.dry_run else 'LIVE (will rename files)'}"
    )
    print()

    if not args.dry_run:
        response = input("⚠️  This will rename files. Continue? [y/N]: ")
        if response.lower() != "y":
            print("Aborted.")
            return

    rename_files(args.base_dir, args.datasets, args.dry_run)


if __name__ == "__main__":
    main()
