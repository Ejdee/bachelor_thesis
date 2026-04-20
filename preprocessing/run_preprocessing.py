#!/usr/bin/env python3
"""
Preprocessing pipeline for CA-ALIKED domain adaptation.

Runs all stages in order on a single vehicle image sequence:
  1. Car mask extraction with 5px dilation  (used for filtering in pseudo-GT)
  2. Car mask extraction with 10px dilation (used for reconstruction)
  3. Apply 10px masks to images             (background removal)
  4. Multi-model SfM reconstruction         (DISK-referenced geometry)
  5. Pseudo ground-truth heatmap generation (point classification)

Usage:
    python run_preprocessing.py --images /path/to/images --output /path/to/output

After this script finishes, the output directory will contain:
    masks/dilate_5/       binary masks with 5px dilation
    masks/dilate_10/      binary masks with 10px dilation
    masked/               background-removed images
    reconstructions/      per-model SfM outputs (aliked, disk, sift, superpoint, r2d2)
    heatmaps/             pseudo-GT heatmaps (.npy) for training
    valid_masks/          training region masks (.npy) for training
"""

import argparse
import shutil
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent


def run_stage(name, cmd):
    import subprocess

    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(
            f"\n[ERROR] Stage '{name}' failed (exit code {result.returncode}). Stopping."
        )
        sys.exit(result.returncode)
    print(f"[OK] {name} done.")


def main():
    parser = argparse.ArgumentParser(
        description="Full preprocessing pipeline: masks → reconstruction → pseudo-GT"
    )
    parser.add_argument(
        "--images", required=True, help="Directory containing the raw input images"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Root output directory (will be created if absent)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID for mask extraction (default: 0)",
    )
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.5,
        help="Quality score threshold for pseudo-GT filtering (default: 0.5)",
    )
    parser.add_argument(
        "--recon-attempts",
        type=int,
        default=10,
        help="Max SfM retry attempts per model (default: 10)",
    )
    args = parser.parse_args()

    images_dir = Path(args.images).resolve()
    output_dir = Path(args.output).resolve()

    if not images_dir.is_dir():
        print(f"[ERROR] Images directory not found: {images_dir}")
        sys.exit(1)

    masks_dir = output_dir / "masks"
    masked_dir = output_dir / "masked"
    recon_dir = output_dir / "reconstructions"

    # ------------------------------------------------------------------
    # Stage 1: Extract 5px masks  (used later for pseudo-GT filtering)
    # ------------------------------------------------------------------
    run_stage(
        "Stage 1/5 — Mask extraction (5px dilation)",
        [
            sys.executable,
            str(SCRIPTS_DIR / "car_mask_extractor.py"),
            "--img",
            str(images_dir),
            "--dilate",
            "5",
            "--mask-dir",
            str(masks_dir / "dilate_5"),
            "--viz-dir",
            str(masks_dir / "dilate_5_viz"),
            "--gpu",
            str(args.gpu),
        ],
    )

    # ------------------------------------------------------------------
    # Stage 2: Extract 10px masks  (used for background removal)
    # ------------------------------------------------------------------
    run_stage(
        "Stage 2/5 — Mask extraction (10px dilation)",
        [
            sys.executable,
            str(SCRIPTS_DIR / "car_mask_extractor.py"),
            "--img",
            str(images_dir),
            "--dilate",
            "10",
            "--mask-dir",
            str(masks_dir / "dilate_10"),
            "--viz-dir",
            str(masks_dir / "dilate_10_viz"),
            "--gpu",
            str(args.gpu),
        ],
    )

    # ------------------------------------------------------------------
    # Stage 3: Apply 10px masks to produce background-free images
    # ------------------------------------------------------------------
    run_stage(
        "Stage 3/5 — Apply masks to images",
        [
            sys.executable,
            str(SCRIPTS_DIR / "apply_masks.py"),
            "--image-dir",
            str(images_dir),
            "--mask-dir",
            str(masks_dir / "dilate_10"),
            "--output-dir",
            str(masked_dir),
        ],
    )

    # ------------------------------------------------------------------
    # Stage 4: Multi-model SfM reconstruction
    # ------------------------------------------------------------------
    run_stage(
        "Stage 4/5 — Multi-model SfM reconstruction",
        [
            sys.executable,
            str(SCRIPTS_DIR / "pipeline_combined.py"),
            "--images",
            str(masked_dir),
            "--output",
            str(recon_dir),
            "--recon_max_attempts",
            str(args.recon_attempts),
        ],
    )

    # ------------------------------------------------------------------
    # Stage 5: Point classification and pseudo-GT heatmap generation
    # ------------------------------------------------------------------
    run_stage(
        "Stage 5/5 — Pseudo-GT heatmap generation",
        [
            sys.executable,
            str(SCRIPTS_DIR / "point_classification.py"),
            "--dataset",
            str(images_dir),
            "--masks",
            str(masks_dir / "dilate_5"),
            "--output",
            str(recon_dir),
            "--output_dir",
            str(output_dir),
            "--score_thresh",
            str(args.score_thresh),
            "--create_heatmaps",
            "--use_masks",
        ],
    )

    # point_classification.py writes masks/ but training expects valid_masks/
    masks_src = output_dir / "masks"
    masks_dst = output_dir / "valid_masks"
    if masks_src.exists():
        if masks_dst.exists():
            shutil.rmtree(masks_dst)
        shutil.move(str(masks_src), str(masks_dst))

    print(f"\n{'=' * 60}")
    print("  All preprocessing stages completed successfully!")
    print(f"  Heatmaps:      {output_dir / 'heatmaps'}")
    print(f"  Training masks:{output_dir / 'valid_masks'}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
