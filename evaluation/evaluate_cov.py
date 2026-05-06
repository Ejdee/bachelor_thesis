#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import pycolmap
import torch
from tqdm import tqdm

# hloc lives in src/hloc/
_HLOC_DIR = Path(__file__).resolve().parents[1] / "hloc"
if str(_HLOC_DIR) not in sys.path:
    sys.path.insert(0, str(_HLOC_DIR))

from hloc import extractors  # noqa: E402
from hloc.utils.base_model import dynamic_load  # noqa: E402

try:
    from safe_gpu import safe_gpu

    safe_gpu.claim_gpus(nb_gpus=1)
except ImportError:
    pass


@dataclass
class ImageEntry:
    name: str
    image_path: Path
    camera: pycolmap.Camera
    cam_from_world: pycolmap.Rigid3d


@dataclass
class ModelSpec:
    key: str
    label: str
    extractor_name: str
    extractor_conf: Dict[str, object]
    grayscale: bool


def parse_thresholds(raw: str) -> List[float]:
    thresholds = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not thresholds:
        raise ValueError("At least one AUC threshold is required.")
    return thresholds


def parse_extractor_list(raw: str) -> List[str]:
    keys = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not keys:
        raise ValueError("At least one extractor key is required.")
    return keys


def resolve_image_path(directory: Path, name: str) -> Path | None:
    """Find an image file by name, falling back to any extension if the exact path is missing."""
    candidate = directory / name
    if candidate.exists():
        return candidate
    matches = sorted(directory.glob(f"{Path(name).stem}.*"))
    return matches[0] if matches else None


def load_entries(image_dir: Path, gt_dir: Path) -> List[ImageEntry]:
    """Load image entries from a COLMAP reconstruction, skipping any images not found on disk."""
    reconstruction = pycolmap.Reconstruction(gt_dir)
    entries: List[ImageEntry] = []
    missing: List[str] = []

    for image in reconstruction.images.values():
        name = Path(image.name).name
        image_path = resolve_image_path(image_dir, name)
        if image_path is None:
            missing.append(name)
            continue
        entries.append(
            ImageEntry(
                name=name,
                image_path=image_path,
                camera=reconstruction.cameras[image.camera_id],
                cam_from_world=image.cam_from_world(),
            )
        )

    if missing:
        print(f"[warn] Missing {len(missing)} images from the image directory.")

    entries.sort(key=lambda item: item.name)
    return entries


def _resize_to_max(image: np.ndarray, max_size: int) -> Tuple[np.ndarray, float]:
    """Downscale an image so its longest side does not exceed max_size, preserving aspect ratio."""
    h, w = image.shape[:2]
    scale = min(1.0, max_size / max(h, w))
    if scale < 1.0:
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return image, scale


def read_image_tensor(
    image_path: Path, grayscale: bool, max_size: int = 3200
) -> Tuple[torch.Tensor, Tuple[int, int], float]:
    """Read an image from disk, optionally downscale it, and return a normalised float tensor."""
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(str(image_path), flag)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    orig_h, orig_w = img.shape[:2]
    img, scale = _resize_to_max(img, max_size)

    if grayscale:
        tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

    return tensor, (orig_h, orig_w), scale


def build_extractor(spec: ModelSpec, device: torch.device):
    """Instantiate and move a feature extractor to the target device, set to eval mode."""
    conf = {"name": spec.extractor_name, **spec.extractor_conf}
    model = dynamic_load(extractors, spec.extractor_name)
    return model(conf).eval().to(device)



def limit_features_by_budget(
    keypoints: np.ndarray,
    descriptors: np.ndarray,
    scores: np.ndarray,
    max_keypoints: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retain only the top-scoring keypoints up to max_keypoints, keeping descriptors in sync."""
    if max_keypoints <= 0 or keypoints.shape[0] <= max_keypoints:
        return keypoints, descriptors, scores

    if scores.shape[0] != keypoints.shape[0] or np.allclose(scores, scores[0]):
        # Scores are missing or all equal -> just take the first N keypoints
        selected = np.arange(max_keypoints, dtype=np.int64)
    else:
        selected = np.argpartition(scores, -max_keypoints)[-max_keypoints:]
        selected = selected[np.argsort(scores[selected])[::-1]]

    # Descriptors can be (N, D) or (D, N) depending on the extractor.
    if descriptors.shape[0] == keypoints.shape[0]:
        desc_sel = descriptors[selected]
    elif descriptors.ndim > 1 and descriptors.shape[1] == keypoints.shape[0]:
        desc_sel = descriptors[:, selected]
    else:
        raise ValueError(
            f"Cannot slice descriptors of shape {descriptors.shape} with {keypoints.shape[0]} keypoints."
        )

    return keypoints[selected], desc_sel, scores[selected]


def descriptors_to_nxd(descriptors: np.ndarray, num_keypoints: int) -> np.ndarray:
    """Ensure descriptors are in (N, D) layout, transposing if the extractor returns (D, N).

    Different extractors use different conventions: some output (N, D) where each
    row is a descriptor, others output (D, N) where each column is a descriptor.
    OpenCV's BFMatcher expects (N, D), so we normalise here.
    """
    if descriptors.ndim != 2:
        raise ValueError(f"Expected 2D descriptors, got {descriptors.shape}")
    if descriptors.shape[0] == num_keypoints:
        return descriptors.astype(np.float32, copy=False)
    if descriptors.shape[1] == num_keypoints:
        return descriptors.T.astype(np.float32, copy=False)
    raise ValueError(
        f"Descriptor shape {descriptors.shape} is incompatible with {num_keypoints} keypoints."
    )


@torch.no_grad()
def extract_features_for_entries(
    spec: ModelSpec,
    extractor,
    entries: Sequence[ImageEntry],
    device: torch.device,
    max_keypoints: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Run the extractor over every image entry and collect keypoints, descriptors, and scores."""
    outputs: Dict[str, Dict[str, np.ndarray]] = {}
    total_keypoints = 0

    for entry in tqdm(entries, desc=f"[{spec.label}] Features", leave=False):
        image_tensor, original_shape, scale = read_image_tensor(
            entry.image_path, grayscale=spec.grayscale
        )
        image_tensor = image_tensor.unsqueeze(0).to(device, non_blocking=True)
        prediction = extractor({"image": image_tensor})

        keypoints = prediction["keypoints"][0].detach().cpu().numpy().astype(np.float64)
        # The extractor ran on a downscaled image, so keypoints are in downscaled
        # pixel coordinates. Divide by scale to bring them back to the original resolution.
        if scale < 1.0:
            keypoints = keypoints / scale

        descriptors = prediction["descriptors"][0].detach().cpu().numpy().astype(np.float32)

        score_tensor = prediction.get("scores", None)
        if score_tensor is None:
            # Not all extractors produce per-keypoint scores (e.g. SIFT)
            scores = np.zeros(keypoints.shape[0], dtype=np.float32)
        else:
            scores = score_tensor[0].detach().cpu().numpy().astype(np.float32).reshape(-1)
        if scores.shape[0] != keypoints.shape[0]:
            # unexpected score shape - fallback to zeros
            scores = np.zeros(keypoints.shape[0], dtype=np.float32)

        keypoints, descriptors, scores = limit_features_by_budget(
            keypoints=keypoints,
            descriptors=descriptors,
            scores=scores,
            max_keypoints=max_keypoints,
        )
        total_keypoints += int(keypoints.shape[0])
        outputs[entry.name] = {
            "keypoints": keypoints,
            "descriptors": descriptors,
            "scores": scores,
            "shape": np.array(original_shape, dtype=np.int32),
        }

    avg = total_keypoints / max(1, len(entries))
    print(f"  -> Total keypoints kept: {total_keypoints} ({avg:.1f} per image)")
    return outputs


def _ratio_pass(
    matcher: cv2.BFMatcher,
    query: np.ndarray,
    train: np.ndarray,
    ratio: float,
) -> Dict[int, int]:
    """Run a single-direction kNN ratio test and return surviving query->train index pairs.

    For each query descriptor the two nearest neighbours in train are found.
    A match is kept only if the closest neighbour is better than
    the second-closest (Lowe's ratio test). This rejects ambiguous matches where
    two descriptors look equally similar.
    """
    result: Dict[int, int] = {}
    for pair in matcher.knnMatch(query, train, k=2):
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
            result[pair[0].queryIdx] = pair[0].trainIdx
    return result


def mutual_ratio_match(
    descriptors_left: np.ndarray,
    descriptors_right: np.ndarray,
    ratio: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Match descriptors with symmetric nearest-neighbour ratio test, keeping only mutual matches.

    Two passes are run: left -> right and right -> left. A pair (i, j) is kept only
    when both passes agree.
    """
    if descriptors_left.shape[0] < 2 or descriptors_right.shape[0] < 2:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = _ratio_pass(matcher, descriptors_left, descriptors_right, ratio)
    backward = _ratio_pass(matcher, descriptors_right, descriptors_left, ratio)

    # Keep only pairs where the reverse match agrees.
    idx_left = [l for l, r in forward.items() if backward.get(r, -1) == l]
    idx_right = [forward[l] for l in idx_left]

    if not idx_left:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    return np.asarray(idx_left, dtype=np.int64), np.asarray(idx_right, dtype=np.int64)


def match_features(
    feat_left: Dict[str, np.ndarray],
    feat_right: Dict[str, np.ndarray],
    ratio: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Match two feature sets and return the corresponding 2-D point arrays."""
    kp_left = feat_left["keypoints"]
    kp_right = feat_right["keypoints"]
    if kp_left.shape[0] < 8 or kp_right.shape[0] < 8:
        return np.empty((0, 2)), np.empty((0, 2))

    desc_left = descriptors_to_nxd(feat_left["descriptors"], kp_left.shape[0])
    desc_right = descriptors_to_nxd(feat_right["descriptors"], kp_right.shape[0])
    idx_left, idx_right = mutual_ratio_match(desc_left, desc_right, ratio=ratio)

    if idx_left.size < 8:
        return np.empty((0, 2)), np.empty((0, 2))
    return kp_left[idx_left], kp_right[idx_right]


def normalize_keypoints(camera: pycolmap.Camera, keypoints: np.ndarray) -> np.ndarray:
    """Unproject pixel coordinates to normalised camera-space rays using the camera model."""
    normalized = np.asarray(camera.cam_from_img(keypoints), dtype=np.float64)
    if normalized.ndim != 2 or normalized.shape[1] != 2:
        raise RuntimeError("camera.cam_from_img did not return an Nx2 array")
    return normalized


def ground_truth_relative_pose(
    left: ImageEntry, right: ImageEntry
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the ground-truth relative rotation and translation from left to right camera."""
    right_from_left = right.cam_from_world * left.cam_from_world.inverse()
    rotation_gt = np.asarray(right_from_left.rotation.matrix(), dtype=np.float64)
    translation_gt = np.asarray(right_from_left.translation, dtype=np.float64)
    return rotation_gt, translation_gt


def angle_error_mat(rotation_1: np.ndarray, rotation_2: np.ndarray) -> float:
    """Compute the geodesic angular distance in degrees between two rotation matrices.

    The relative rotation R_rel = R1^T @ R2 maps from frame 2 into frame 1.
    Its rotation angle is recovered via the trace identity:
        cos(angle) = (trace(R_rel) - 1) / 2
    which comes from the fact that for any rotation matrix, the trace equals
    1 + 2*cos(angle). We clip to [-1, 1] to guard against floating-point drift.
    """
    cos_angle = (np.trace(rotation_1.T @ rotation_2) - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.abs(np.arccos(cos_angle))))


def angle_error_vec(vector_1: np.ndarray, vector_2: np.ndarray) -> float:
    """Compute the angular difference in degrees between two direction vectors.

    Used for the translation error: the recovered translation is only known up
    to scale and sign, so the caller handles the 180° ambiguity separately.
    Returns inf if either vector is near-zero (degenerate case).
    """
    norm = np.linalg.norm(vector_1) * np.linalg.norm(vector_2)
    if norm <= 1e-12:
        return float("inf")
    cos_angle = np.clip(np.dot(vector_1, vector_2) / norm, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def estimate_pose_error_from_points(
    left: ImageEntry,
    right: ImageEntry,
    points_left: np.ndarray,
    points_right: np.ndarray,
    fundamental_threshold_px: float,
    ransac_confidence: float,
    ransac_max_iters: int,
) -> float:
    """Estimate relative pose via F-RANSAC and return the max of rotation and translation errors in degrees.

    Pipeline:
      1. Normalise pixel coordinates using camera intrinsics (undistort + unproject).
      2. Run RANSAC to estimate the fundamental matrix F in pixel space.
      3. Convert F to the essential matrix E using the calibration matrices K.
      4. Decompose E into (R, t) via recoverPose, which also resolves the sign ambiguity.
      5. Compare against the ground-truth relative pose and return the larger of the
         rotation error and translation error (the standard IMC benchmark metric).
    """
    if points_left.shape[0] < 8 or points_right.shape[0] < 8:
        return float("inf")

    # Step 1 - normalise pixels to camera-space rays using the camera models.
    # Points with non-finite coordinates after undistortion are discarded.
    try:
        points_left_normalized = normalize_keypoints(left.camera, points_left)
        points_right_normalized = normalize_keypoints(right.camera, points_right)
    except Exception:
        return float("inf")

    valid = (
        np.isfinite(points_left_normalized).all(axis=1)
        & np.isfinite(points_right_normalized).all(axis=1)
    )
    points_left = points_left[valid]
    points_right = points_right[valid]
    points_left_normalized = points_left_normalized[valid]
    points_right_normalized = points_right_normalized[valid]

    if points_left_normalized.shape[0] < 8:
        return float("inf")

    # Step 2 - estimate the fundamental matrix F with RANSAC.
    # F maps a point in the left image to its epipolar line in the right image.
    # We use the pixel coordinates here because the threshold is in pixels.
    # The outer try/except handles older OpenCV versions that don't accept maxIters.
    try:
        fundamental, mask = cv2.findFundamentalMat(
            points_left,
            points_right,
            method=cv2.FM_RANSAC,
            ransacReprojThreshold=fundamental_threshold_px,
            confidence=min(ransac_confidence, 0.999999),
            maxIters=ransac_max_iters,
        )
    except TypeError:
        try:
            fundamental, mask = cv2.findFundamentalMat(
                points_left,
                points_right,
                method=cv2.FM_RANSAC,
                ransacReprojThreshold=fundamental_threshold_px,
                confidence=min(ransac_confidence, 0.999999),
            )
        except cv2.error:
            return float("inf")
    except cv2.error:
        return float("inf")

    if fundamental is None or mask is None:
        return float("inf")

    # OpenCV can return multiple stacked 3x3 matrices when the scene is degenerate.
    # We collect all candidates and pick the best one below.
    if fundamental.shape == (3, 3):
        fundamental_candidates = [fundamental]
    elif fundamental.shape[1] == 3 and fundamental.shape[0] % 3 == 0:
        fundamental_candidates = [
            fundamental[i: i + 3] for i in range(0, fundamental.shape[0], 3)
        ]
    else:
        return float("inf")

    inlier_mask = mask.astype(bool).reshape(-1)
    if inlier_mask.sum() < 5:
        return float("inf")

    # Step 3 - convert F to the essential matrix E.
    # The relationship is:  E = K_right^T @ F @ K_left
    # E encodes the same epipolar geometry but in normalised (calibrated) coordinates,
    # which allows decomposing it into a pure rotation + translation.
    K_left = np.asarray(left.camera.calibration_matrix(), dtype=np.float64)
    K_right = np.asarray(right.camera.calibration_matrix(), dtype=np.float64)

    best_rotation = None
    best_translation = None
    best_inlier_count = -1

    for F in fundamental_candidates:
        essential = K_right.T @ F @ K_left

        # Step 4 - decompose E into (R (rotation), t (translation)).
        # recoverPose tries the four possible (R, t) solutions and picks the one
        # where the most inlier points project in front of both cameras (positive depth).
        # Normalised coordinates are passed so we use identity as the camera matrix.
        try:
            inlier_count, R_est, t_est, _ = cv2.recoverPose(
                essential,
                points_left_normalized[inlier_mask],
                points_right_normalized[inlier_mask],
                cameraMatrix=np.eye(3),
            )
        except Exception:
            continue
        if inlier_count > best_inlier_count:
            best_inlier_count = int(inlier_count)
            best_rotation = R_est
            best_translation = t_est.reshape(-1)

    if best_rotation is None or best_translation is None:
        return float("inf")

    # Step 5 - compare estimated pose to ground truth.
    rotation_gt, translation_gt = ground_truth_relative_pose(left, right)
    rotation_error = angle_error_mat(best_rotation, rotation_gt)
    translation_error = angle_error_vec(best_translation, translation_gt)
    # Translation is only known up to scale, and recoverPose can return the
    # direction or its opposite. Taking min(e, 180-e) handles the sign ambiguity.
    if math.isfinite(translation_error):
        translation_error = min(translation_error, 180.0 - translation_error)

    total_error = max(rotation_error, translation_error)
    return float(total_error) if math.isfinite(total_error) else float("inf")


def pose_auc(errors: Sequence[float], thresholds: Sequence[float]) -> List[float]:
    """Compute the normalised AUC of the pose-error recall curve at each given threshold.

    The recall curve maps each error threshold t to the fraction of pairs whose
    pose error is <= t. The AUC is the area under that curve from 0 to the given
    threshold, divided by the threshold so the result is in [0, 1].

    Failed pairs (inf error) count against recall but are excluded from the sorted
    array, so they simply reduce the curve's height without distorting its shape.
    """
    total_pairs = len(errors)
    values = np.sort(np.asarray(errors, dtype=np.float64))
    # Infinite errors (failed pairs) are excluded from the curve but their count
    # is kept in total_pairs so they lower the recall fraction.
    values = values[np.isfinite(values)]

    if total_pairs == 0:
        return [0.0 for _ in thresholds]

    # Build the recall curve: after sorting, the i-th error value corresponds to
    # a recall of (i+1) / total_pairs (fraction of pairs at or below that error).
    recall = (np.arange(values.size, dtype=np.float64) + 1.0) / float(total_pairs)
    # Prepend (0, 0) so the curve starts at the origin.
    values = np.r_[0.0, values]
    recall = np.r_[0.0, recall]

    aucs: List[float] = []
    for threshold in thresholds:
        # Find where the error axis would exceed the threshold.
        last = int(np.searchsorted(values, threshold, side="right"))
        if last <= 0:
            aucs.append(0.0)
            continue
        # Clip the curve at the threshold and integrate with the trapezoid rule.
        # The final point is interpolated by repeating the last recall value up to
        # the threshold (recall is flat once all finite errors are accounted for).
        aucs.append(
            float(
                np.trapezoid(
                    np.r_[recall[:last], recall[last - 1]],
                    x=np.r_[values[:last], threshold],
                )
                / threshold
            )
        )
    return aucs



def make_all_pairs(names: Sequence[str]) -> List[Tuple[str, str]]:
    """Return all unique unordered image-name pairs."""
    return list(itertools.combinations(names, 2))


def make_model_specs(
    selected_keys: Sequence[str], aliked_model_name: str, fine_weights: Path
) -> List[ModelSpec]:
    """Build ModelSpec objects for the requested extractor keys."""
    all_specs: Dict[str, ModelSpec] = {
        "sift": ModelSpec(
            key="sift",
            label="SIFT",
            extractor_name="dog",
            extractor_conf={
                "descriptor": "rootsift",
                "max_keypoints": -1,
                "options": {"first_octave": 0, "peak_threshold": 0.01},
            },
            grayscale=True,
        ),
        "superpoint": ModelSpec(
            key="superpoint",
            label="SuperPoint",
            extractor_name="superpoint",
            extractor_conf={
                "nms_radius": 4,
                "keypoint_threshold": 0.005,
                "max_keypoints": 5000,
            },
            grayscale=True,
        ),
        "r2d2": ModelSpec(
            key="r2d2",
            label="R2D2",
            extractor_name="r2d2",
            extractor_conf={
                "model_name": "r2d2_WASF_N16.pt",
                "max_keypoints": 5000,
                "reliability_threshold": 0.7,
                "repetability_threshold": 0.7,
            },
            grayscale=False,
        ),
        "aliked_base": ModelSpec(
            key="aliked_base",
            label="ALIKED",
            extractor_name="aliked",
            extractor_conf={
                "model_name": aliked_model_name,
                "max_num_keypoints": 5000,
                "detection_threshold": 0.1,
                "nms_radius": 2,
            },
            grayscale=False,
        ),
        "aliked_tuned": ModelSpec(
            key="aliked_tuned",
            label="CA-ALIKED",
            extractor_name="aliked",
            extractor_conf={
                "model_name": aliked_model_name,
                "max_num_keypoints": 5000,
                "detection_threshold": 0.1,
                "nms_radius": 2,
                "weights": str(fine_weights),
            },
            grayscale=False,
        ),
        "disk": ModelSpec(
            key="disk",
            label="DISK",
            extractor_name="disk",
            extractor_conf={
                "weights": "depth",
                "max_keypoints": 5000,
                "detection_threshold": 0.0,
                "nms_window_size": 5,
            },
            grayscale=False,
        ),
    }

    specs: List[ModelSpec] = []
    for key in selected_keys:
        if key not in all_specs:
            raise ValueError(
                f"Unknown extractor key '{key}'. Available: {', '.join(all_specs.keys())}"
            )
        specs.append(all_specs[key])
    return specs


def print_comparison_table(
    thresholds: Sequence[float], results: Dict[str, Dict[str, object]]
) -> None:
    """Print a formatted AUC comparison table to stdout."""
    headers = [
        f"AUC@{int(v) if float(v).is_integer() else v}" for v in thresholds
    ]
    header_row = (
        "Model".ljust(22) + " | " + " | ".join(h.center(10) for h in headers) + " | Success"
    )
    sep = "-" * len(header_row)

    print("\n=== Relative Pose AUC Comparison ===")
    print(sep)
    print(header_row)
    print(sep)
    for label, data in results.items():
        errors = data["errors"]
        success = int(np.isfinite(np.asarray(errors)).sum())
        values = [f"{float(s) * 100.0:8.2f}%" for s in data["aucs"]]
        print(label.ljust(22) + " | " + " | ".join(values) + f" | {success}/{len(errors)}")
    print(sep)


def evaluate_model(
    spec: ModelSpec,
    extractor,
    entries: Sequence[ImageEntry],
    pairs: Sequence[Tuple[str, str]],
    device: torch.device,
    fallback_ratio: float,
    fundamental_threshold_px: float,
    ransac_confidence: float,
    ransac_max_iters: int,
    max_keypoints: int,
) -> List[float]:
    """Extract features, match all pairs, estimate poses, and return per-pair angular errors."""
    print(f"\n[{spec.label}] Extracting features for {len(entries)} images...")
    features_by_name = extract_features_for_entries(
        spec=spec,
        extractor=extractor,
        entries=entries,
        device=device,
        max_keypoints=max_keypoints,
    )

    entries_by_name = {entry.name: entry for entry in entries}
    errors: List[float] = []

    for left_name, right_name in tqdm(pairs, desc=f"[{spec.label}] Pairs", leave=False):
        points_left, points_right = match_features(
            features_by_name[left_name], features_by_name[right_name], ratio=fallback_ratio
        )
        error = estimate_pose_error_from_points(
            left=entries_by_name[left_name],
            right=entries_by_name[right_name],
            points_left=points_left,
            points_right=points_right,
            fundamental_threshold_px=fundamental_threshold_px,
            ransac_confidence=ransac_confidence,
            ransac_max_iters=ransac_max_iters,
        )
        errors.append(error)

    successful = int(np.isfinite(np.asarray(errors)).sum())
    print(f"[{spec.label}] Successful pose estimates: {successful}/{len(errors)}")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate local feature extractors with the IMC paper protocol: exhaustive pairs, "
            "symmetric NN-ratio matching, and fundamental-matrix RANSAC pose recovery."
        )
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("/zfs-pool/home/xbehoua00/filtered_images/200-400/2024_04_12_11_49_59/"),
        help="Directory containing the images to evaluate.",
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=Path("/zfs-pool/home/xbehoua00/zms-tool/xbehoua00/train/ALIKED/sift-rec-ref/sift/sfm_best/"),
        help="COLMAP reconstruction directory containing cameras/images model files.",
    )
    parser.add_argument(
        "--fine-weights",
        type=Path,
        default=Path("/zfs-pool/home/xbehoua00/recs/aliked_1080p_punisher_ep10_end.pth"),
        help="Path to fine-tuned ALIKED .pth weights.",
    )
    parser.add_argument(
        "--extractors",
        type=str,
        default="sift,superpoint,r2d2,aliked_base,aliked_tuned,disk",
        help="Comma-separated extractor keys to evaluate.",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="5,10,20",
        help="Comma-separated angular thresholds in degrees for AUC.",
    )
    parser.add_argument(
        "--fallback-ratio",
        type=float,
        default=0.9,
        help="Lowe ratio threshold for symmetric nearest-neighbour matching.",
    )
    parser.add_argument(
        "--fundamental-threshold-px",
        type=float,
        default=0.5,
        help="RANSAC inlier threshold in pixels for fundamental matrix estimation.",
    )
    parser.add_argument(
        "--ransac-confidence",
        type=float,
        default=0.999999,
        help="RANSAC confidence used by fundamental-matrix estimation.",
    )
    parser.add_argument(
        "--ransac-max-iters",
        type=int,
        default=100000,
        help="Maximum RANSAC iterations for fundamental-matrix estimation.",
    )
    parser.add_argument(
        "--max-keypoints",
        type=int,
        default=5000,
        help="Global maximum number of keypoints per image.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Execution device. 'auto' picks CUDA when available.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("evaluate_cov_metrics.json"),
        help="Where to write the summary metrics JSON.",
    )
    return parser.parse_args()


def main() -> None:
    """Parse arguments, run evaluation for each extractor, and write results to JSON and plots."""
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds)
    selected_extractors = parse_extractor_list(args.extractors)

    if not args.image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {args.image_dir}")
    if not args.gt_dir.exists():
        raise FileNotFoundError(f"Ground-truth reconstruction directory not found: {args.gt_dir}")
    if not args.fine_weights.exists():
        raise FileNotFoundError(f"Fine-tuned weights not found: {args.fine_weights}")
    if args.max_keypoints <= 0:
        raise ValueError(f"--max-keypoints must be > 0, got {args.max_keypoints}")

    model_specs = make_model_specs(selected_extractors, "aliked-n32", args.fine_weights)
    entries = load_entries(args.image_dir, args.gt_dir)
    if len(entries) < 2:
        raise RuntimeError("Need at least 2 images shared between the image directory and GT model.")

    pairs = make_all_pairs([entry.name for entry in entries])
    print(f"Loaded {len(entries)} images with GT poses.")
    print(f"Evaluating {len(pairs)} image pairs.")
    print("Evaluating extractors: " + ", ".join(spec.key for spec in model_specs))
    print(
        f"Protocol: IMC paper-style evaluation "
        f"(symmetric NN-ratio, ratio={args.fallback_ratio}, F-RANSAC, th={args.fundamental_threshold_px}px)"
    )

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results: Dict[str, Dict[str, object]] = {}
    for spec in tqdm(model_specs, desc="Models", leave=False):
        extractor = build_extractor(spec, device)
        errors = evaluate_model(
            spec=spec,
            extractor=extractor,
            entries=entries,
            pairs=pairs,
            device=device,
            fallback_ratio=args.fallback_ratio,
            fundamental_threshold_px=args.fundamental_threshold_px,
            ransac_confidence=args.ransac_confidence,
            ransac_max_iters=args.ransac_max_iters,
            max_keypoints=args.max_keypoints,
        )
        results[spec.label] = {"key": spec.key, "errors": errors, "aucs": pose_auc(errors, thresholds)}
        del extractor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print_comparison_table(thresholds, results)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "thresholds": thresholds,
        "models": {
            label: {
                "key": data["key"],
                "aucs": data["aucs"],
                "total_pairs": len(data["errors"]),
                "successful_pairs": int(np.isfinite(np.asarray(data["errors"])).sum()),
            }
            for label, data in results.items()
        },
    }
    with args.output_json.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved metrics to: {args.output_json}")


if __name__ == "__main__":
    main()
