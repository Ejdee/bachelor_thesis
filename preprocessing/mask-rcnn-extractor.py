"""=============================================================================
Project:     Accurate and Robust Localization of Landmarks on a Vehicle
Author:      Adam Běhoun <xbehoua00@vutbr.cz>
Year:        2026
Description: Extracts car masks with Detectron2.
============================================================================="""

import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(
    description="Extract car instance masks using Mask R-CNN"
)
parser.add_argument(
    "--img", type=str, required=True, help="Path to an image or a directory of images"
)
parser.add_argument(
    "--dilate", type=int, default=3, help="Dilation kernel size in pixels (default: 3)"
)
parser.add_argument(
    "--pad",
    type=int,
    default=0,
    help="Padding to add around the image before prediction (default: 0)",
)
parser.add_argument(
    "--mask-dir",
    type=str,
    default="masks_output",
    help="Directory to save binary mask images (default: masks_output)",
)
parser.add_argument(
    "--viz-dir",
    type=str,
    default="masks_viz",
    help="Directory to save visualization overlays (default: masks_viz)",
)
parser.add_argument(
    "--gpu", type=int, default=0, help="GPU device ID (-1 for CPU, default: 0)"
)
args = parser.parse_args()

cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
)
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

if args.gpu >= 0:
    cfg.MODEL.DEVICE = f"cuda:{args.gpu}"
    print(f"[INFO] Using GPU: cuda:{args.gpu}")
else:
    cfg.MODEL.DEVICE = "cpu"
    print("[INFO] Using CPU")

predictor = DefaultPredictor(cfg)

if os.path.isdir(args.img):
    image_files = sorted(
        [
            os.path.join(args.img, f)
            for f in os.listdir(args.img)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    )
else:
    image_files = [args.img]

if not image_files:
    print("[ERROR] No images found")
    exit(1)

print(f"[INFO] Found {len(image_files)} image(s)")
os.makedirs(args.mask_dir, exist_ok=True)
os.makedirs(args.viz_dir, exist_ok=True)

CAR_CLASS_ID = 2

for img_path in image_files:
    image = cv2.imread(img_path)
    if image is None:
        print(f"[WARN] Failed to read: {img_path}")
        continue

    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")
    masks = instances.pred_masks.numpy()
    classes = instances.pred_classes.numpy()

    car_masks = masks[classes == CAR_CLASS_ID]

    # Select the largest car mask if multiple are detected
    if car_masks.size > 0:
        areas = [np.sum(m) for m in car_masks]
        largest_idx = np.argmax(areas)
        mask = car_masks[largest_idx].astype(np.uint8)
        print(
            f"{os.path.basename(img_path)}: {len(car_masks)} car(s) detected, selected largest ({areas[largest_idx]} px)"
        )
    else:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        print(f"{os.path.basename(img_path)}: [WARN] no car detected")

    kernel = np.ones((args.dilate, args.dilate), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)

    stem = os.path.splitext(os.path.basename(img_path))[0]
    mask_out = os.path.join(args.mask_dir, f"{stem}_mask.png")
    cv2.imwrite(mask_out, dilated_mask * 255)

    # Visualization overlay
    viz = image.copy()
    viz[dilated_mask > 0] = (
        viz[dilated_mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
    ).astype(np.uint8)
    cv2.imwrite(os.path.join(args.viz_dir, f"{stem}_viz.png"), viz)

print(f"\n[INFO] Done. Saved {len(image_files)} masks to {args.mask_dir}")
