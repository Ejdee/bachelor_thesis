import argparse
import sys
from pathlib import Path

import cv2


def apply_mask(image_path, mask_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    _, mask_bin = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

    if image.shape[:2] != mask_bin.shape:
        raise ValueError(
            f"Shape mismatch for {image_path.name}: "
            f"image {image.shape[:2]} vs mask {mask_bin.shape}"
        )

    return cv2.bitwise_and(image, image, mask=mask_bin)


def main():
    parser = argparse.ArgumentParser(
        description="Apply binary masks to images and save the masked results"
    )
    parser.add_argument(
        "--image-dir", required=True, help="Directory containing the original images"
    )
    parser.add_argument(
        "--mask-dir",
        required=True,
        help="Directory containing the mask images (*_mask.png)",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory to write the masked images"
    )
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p
        for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    if not image_paths:
        print(f"[WARN] No images found in {image_dir}")
        sys.exit(0)

    ok, skipped = 0, 0
    for image_path in image_paths:
        mask_path = mask_dir / f"{image_path.stem}_mask.png"
        try:
            masked = apply_mask(image_path, mask_path)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[WARN] Skipping {image_path.name}: {exc}")
            skipped += 1
            continue

        out_path = output_dir / image_path.name
        if cv2.imwrite(str(out_path), masked):
            ok += 1
        else:
            print(f"[WARN] Failed to write {out_path}")
            skipped += 1

    print(f"[INFO] Done. {ok} images written, {skipped} skipped.")


if __name__ == "__main__":
    main()
