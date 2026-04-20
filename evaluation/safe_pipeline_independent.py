from pathlib import Path
import copy
import pycolmap
import shutil
import sys
import argparse

# hloc lives in final/hloc/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hloc"))
from hloc import extract_features, match_features, reconstruction, pairs_from_exhaustive

try:
    from safe_gpu import safe_gpu

    safe_gpu.claim_gpus(nb_gpus=1)
    print("[GPU] Successfully claimed 1 GPU")
except ImportError:
    pass

# Extractor and matcher to run. Add more entries to evaluate additional models.
MODELS = {
    "aliked_custom_lg": ["aliked-custom-nn", "aliked+lightglue"],
}

parser = argparse.ArgumentParser(
    description="Single-model SfM reconstruction pipeline (evaluation)"
)
parser.add_argument("--images", required=True, help="Directory of input images")
parser.add_argument("--output", required=True, help="Root output directory")
parser.add_argument(
    "--aliked_weights", default=None, help="Path to fine-tuned ALIKED weights (.pth)"
)
parser.add_argument(
    "--recon_max_attempts", type=int, default=1, help="Max SfM retries (default: 1)"
)
args = parser.parse_args()

# Setup paths
images = Path(args.images)
images_list = sorted(list(images.glob("*.png")) + list(images.glob("*.jpg")))
out_dir = Path(args.output)
out_dir.mkdir(parents=True, exist_ok=True)


def count_registered_images(recon_obj):
    if recon_obj is None:
        return 0
    return len(recon_obj.images)


for model_name in MODELS:
    print(f"\n=== Processing Model: {model_name} ===")

    model_output_dir = out_dir / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # Standard HLOC paths
    feature_path = model_output_dir / "features.h5"
    matcher_path = model_output_dir / "matches.h5"
    pairs_path = model_output_dir / "pairs-sfm.txt"
    reconstruction_path = model_output_dir / "sfm"
    final_output_path = model_output_dir / "sfm-triangulation"

    if feature_path.exists():
        feature_path.unlink()
    if matcher_path.exists():
        matcher_path.unlink()

    with open(model_output_dir / "images_list.txt", "w") as f:
        for img in images_list:
            f.write(img.name + "\n")

    print(f"[{model_name}] Extracting Features...")
    feature_conf = extract_features.confs[MODELS[model_name][0]]
    if args.aliked_weights and "aliked-custom" in str(feature_conf):
        feature_conf = copy.deepcopy(feature_conf)
        feature_conf["model"]["weights"] = args.aliked_weights

    extract_features.main(feature_conf, images, feature_path=feature_path)

    print(f"[{model_name}] Matching Features...")
    pairs_from_exhaustive.main(
        pairs_path, image_list=model_output_dir / "images_list.txt"
    )
    matcher_conf = match_features.confs[MODELS[model_name][1]]
    match_features.main(
        matcher_conf, pairs_path, features=feature_path, matches=matcher_path
    )

    print(
        f"[{model_name}] Starting Reconstruction Loop (Max {args.recon_max_attempts} attempts)..."
    )

    best_num_reg = -1
    best_recon_path = model_output_dir / "sfm_best"

    if best_recon_path.exists():
        shutil.rmtree(best_recon_path)

    image_names_list = [p.name for p in images_list]
    total_images = len(image_names_list)

    for attempt in range(1, args.recon_max_attempts + 1):
        print(f"   > Attempt {attempt}/{args.recon_max_attempts}...")

        if reconstruction_path.exists():
            shutil.rmtree(reconstruction_path)

        try:
            model_recon = reconstruction.main(
                reconstruction_path,
                images,
                pairs_path,
                feature_path,
                matcher_path,
                image_list=image_names_list,
            )
        except Exception as e:
            print(f"     [Error] Attempt {attempt} crashed: {e}")
            model_recon = None

        num_reg = count_registered_images(model_recon)
        print(f"     Result: {num_reg}/{total_images} images registered.")

        if num_reg > best_num_reg:
            print(f"     [New Best] Found better model with {num_reg} images.")
            best_num_reg = num_reg

            if best_recon_path.exists():
                shutil.rmtree(best_recon_path)
            shutil.copytree(reconstruction_path, best_recon_path)

        if num_reg == total_images:
            print(f"     [Success] Perfect reconstruction found! Stopping loop.")
            break

    if best_num_reg > 0:
        print(
            f"[{model_name}] Saving best model ({best_num_reg} images) to final output..."
        )

        if final_output_path.exists():
            shutil.rmtree(final_output_path)

        shutil.copytree(best_recon_path, final_output_path)
        print(f"[{model_name}] Success. Model saved to {final_output_path}")
    else:
        print(f"[{model_name}] Failed to reconstruct anything.")
        if final_output_path.exists():
            shutil.rmtree(final_output_path)
        final_output_path.mkdir()

    print(f"[{model_name}] Done.\n")
