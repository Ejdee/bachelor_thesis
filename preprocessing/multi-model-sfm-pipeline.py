"""=============================================================================
Project:     Accurate and Robust Localization of Landmarks on a Vehicle
Author:      Adam Běhoun <xbehoua00@vutbr.cz>
Year:        2026
Description: SfM pipeline that runs multiple models with retries and selects the best reconstruction for each, then triangulates all models against a shared reference geometry.
============================================================================="""

from pathlib import Path
import copy
import pycolmap
import shutil
import sys
import argparse
import os

# hloc lives one level up from this script
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hloc"))
from hloc import (
    extract_features,
    match_features,
    reconstruction,
    pairs_from_exhaustive,
    triangulation,
)

MODELS = {
    "superpoint": ["superpoint_aachen", "superpoint+lightglue"],
    "aliked": ["aliked-n16", "aliked+lightglue"],
    "disk": ["disk", "disk+lightglue"],
    "sift": ["sift", "NN-ratio"],
    "r2d2": ["r2d2", "NN-ratio"],
}


def count_registered_images(recon_path):
    """Count the number of registered images in a reconstruction."""
    try:
        return len(pycolmap.Reconstruction(recon_path).images)
    except Exception:
        return 0


def get_sfm_candidates(model_path):
    """Return sfm_best itself plus any numbered sub-models inside sfm_best/models/."""
    normalized = os.path.normpath(model_path)
    parent_dir = os.path.dirname(normalized)
    sfm_best = (
        os.path.dirname(parent_dir)
        if os.path.basename(parent_dir) == "models"
        else normalized
    )

    candidates = [sfm_best]
    models_dir = os.path.join(sfm_best, "models")
    if os.path.isdir(models_dir):
        children = sorted(
            [
                os.path.join(models_dir, c)
                for c in os.listdir(models_dir)
                if os.path.isdir(os.path.join(models_dir, c))
            ],
            key=lambda p: (
                0 if os.path.basename(p).isdigit() else 1,
                (
                    int(os.path.basename(p))
                    if os.path.basename(p).isdigit()
                    else os.path.basename(p)
                ),
            ),
        )
        for child in children:
            if child not in candidates:
                candidates.append(child)
    return candidates


def load_best_model(model_path):
    """Return the candidate path with the most 3D points, plus the point count."""
    best_path, best_count = None, 0
    for candidate in get_sfm_candidates(model_path):
        try:
            count = len(pycolmap.Reconstruction(candidate).points3D)
        except Exception:
            count = 0
        if count > best_count:
            best_count = count
            best_path = candidate
    return best_path, best_count


def filter_pairs_to_reconstruction(pairs_path, reconstruction_path):
    """Keep only pairs where both images are present in the reconstruction."""
    reference = pycolmap.Reconstruction(reconstruction_path)
    image_names = {Path(img.name).name.lower() for img in reference.images.values()}
    filtered_path = pairs_path.with_name("pairs-sfm-registered.txt")

    kept, total = 0, 0
    with open(pairs_path) as src, open(filtered_path, "w") as dst:
        for line in src:
            total += 1
            n0, n1 = line.split()
            if (
                Path(n0).name.lower() in image_names
                and Path(n1).name.lower() in image_names
            ):
                dst.write(line)
                kept += 1

    print(f"  Filtered pairs: {kept}/{total} kept → {filtered_path.name}")
    return filtered_path, kept, total


def main():
    parser = argparse.ArgumentParser(
        description="Multi-model SfM reconstruction pipeline"
    )
    parser.add_argument(
        "--images", required=True, help="Directory of masked input images"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Root output directory for all model reconstructions",
    )
    parser.add_argument(
        "--aliked_weights",
        default=None,
        help="Optional path to fine-tuned ALIKED weights (.pth)",
    )
    parser.add_argument(
        "--recon_max_attempts",
        type=int,
        default=10,
        help="Maximum SfM retry attempts per model (default: 10)",
    )
    args = parser.parse_args()

    images = Path(args.images)
    images_list = sorted(list(images.glob("*.png")) + list(images.glob("*.jpg")))
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for model in MODELS:
        print(f"\n=== Processing Model: {model} ===")
        model_dir = out_dir / model
        model_dir.mkdir(parents=True, exist_ok=True)

        images_list_path = model_dir / "images_list.txt"
        with open(images_list_path, "w") as f:
            for p in images_list:
                f.write(p.name + "\n")

        feature_path = model_dir / "features.h5"
        matcher_path = model_dir / "matches.h5"
        pairs_path = model_dir / "pairs-sfm.txt"
        reconstruction_path = model_dir / "sfm"
        triangulation_path = model_dir / "sfm-triangulation"
        best_recon_path = model_dir / "sfm_best"

        for stale in [feature_path, matcher_path]:
            if stale.exists():
                stale.unlink()
        if best_recon_path.exists():
            shutil.rmtree(best_recon_path)

        feature_conf = extract_features.confs[MODELS[model][0]]
        if args.aliked_weights:
            model_conf = feature_conf.get("model", {})
            is_aliked = (
                model_conf.get("name") == "aliked"
                or str(model_conf.get("model_name", "")).startswith("aliked")
                or "aliked" in str(feature_conf.get("output", ""))
            )
            if is_aliked:
                feature_conf = copy.deepcopy(feature_conf)
                feature_conf["model"]["weights"] = args.aliked_weights
                print(f"  Using ALIKED weights: {args.aliked_weights}")

        matcher_conf = match_features.confs[MODELS[model][1]]

        print(f"[{model}] Extracting features...")
        extract_features.main(feature_conf, images, feature_path=feature_path)

        print(f"[{model}] Generating image pairs...")
        pairs_from_exhaustive.main(pairs_path, image_list=images_list_path)

        print(f"[{model}] Matching features...")
        match_features.main(
            matcher_conf, pairs_path, features=feature_path, matches=matcher_path
        )

        # SfM with retry loop — keep the best reconstruction found
        image_names = [p.name for p in images_list]
        total_images = len(image_names)
        best_num_reg = -1
        filtered_pairs_path = None

        for attempt in range(1, args.recon_max_attempts + 1):
            print(f"  Attempt {attempt}/{args.recon_max_attempts}...")
            if reconstruction_path.exists():
                shutil.rmtree(reconstruction_path, ignore_errors=True)

            try:
                reconstruction.main(
                    reconstruction_path,
                    images,
                    pairs_path,
                    feature_path,
                    matcher_path,
                    image_list=image_names,
                )
            except Exception as e:
                print(f"    Reconstruction crashed: {e}")

            num_reg = count_registered_images(reconstruction_path)
            print(f"    Registered: {num_reg}/{total_images} images")

            if num_reg > best_num_reg:
                best_num_reg = num_reg
                if best_recon_path.exists():
                    shutil.rmtree(best_recon_path)
                if reconstruction_path.exists():
                    shutil.copytree(reconstruction_path, best_recon_path)

            try:
                filtered_pairs_path, kept, total = filter_pairs_to_reconstruction(
                    pairs_path, reconstruction_path
                )
            except Exception as e:
                print(f"    Pair filtering failed: {e}")
                filtered_pairs_path = None

            if num_reg == total_images:
                print("    Perfect reconstruction — stopping early.")
                break
            if filtered_pairs_path is not None and kept == total:
                print("    Pairs fully consistent — stopping early.")
                break

        final_recon_path = (
            best_recon_path if best_recon_path.exists() else reconstruction_path
        )
        print(
            f"[{model}] Best reconstruction: {best_num_reg}/{total_images} images registered."
        )

        results[model] = {
            "model_dir": model_dir,
            "triangulation_path": triangulation_path,
            "pairs_path": filtered_pairs_path if filtered_pairs_path else pairs_path,
            "feature_path": feature_path,
            "matcher_path": matcher_path,
            "final_recon_path": final_recon_path,
            "best_num_reg": best_num_reg,
        }

    # ------------------------------------------------------------------
    # Find DISK's best reconstruction to use as shared geometry reference
    # ------------------------------------------------------------------
    print("\n=== Finding DISK geometry reference ===")
    disk_ref_path, disk_point_count = load_best_model(
        str(results["disk"]["final_recon_path"])
    )
    print(f"  DISK reference: {disk_ref_path} ({disk_point_count} points)")

    # ------------------------------------------------------------------
    # Triangulate every model against the DISK camera poses
    # ------------------------------------------------------------------
    print("\n=== Triangulation Phase ===")
    for model_name, meta in results.items():
        tri_path = meta["triangulation_path"]
        pairs = meta["pairs_path"]
        features = meta["feature_path"]
        matches = meta["matcher_path"]
        ref = Path(disk_ref_path) if disk_ref_path else meta["final_recon_path"]

        print(
            f"[{model_name}] Triangulating against DISK geometry ({disk_point_count} pts)..."
        )
        if tri_path.exists():
            shutil.rmtree(tri_path, ignore_errors=True)

        triangulation.main(tri_path, ref, images, pairs, features, matches)
        print(f"[{model_name}] Done → {tri_path}\n")


if __name__ == "__main__":
    main()
