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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pycolmap
import torch
from cycler import cycler
from tqdm import tqdm

# hloc lives in final/hloc/
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
    candidate = directory / name
    if candidate.exists():
        return candidate

    stem = Path(name).stem
    matches = sorted(directory.glob(f"{stem}.*"))
    if matches:
        return matches[0]
    return None


def load_entries(image_dir: Path, gt_dir: Path) -> List[ImageEntry]:
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


def read_image_tensor(
    image_path: Path, grayscale: bool, max_size: int = 3200
) -> Tuple[torch.Tensor, Tuple[int, int], float]:
    if grayscale:
        image_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image_gray is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        original_height, original_width = image_gray.shape[:2]
        scale = min(1.0, max_size / max(original_height, original_width))
        if scale < 1.0:
            new_height = int(original_height * scale)
            new_width = int(original_width * scale)
            image_gray = cv2.resize(
                image_gray, (new_width, new_height), interpolation=cv2.INTER_AREA
            )
        image_tensor = torch.from_numpy(image_gray).float().unsqueeze(0) / 255.0
        return image_tensor, (original_height, original_width), scale

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    original_height, original_width = image_bgr.shape[:2]
    scale = min(1.0, max_size / max(original_height, original_width))
    if scale < 1.0:
        new_height = int(original_height * scale)
        new_width = int(original_width * scale)
        image_bgr = cv2.resize(
            image_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA
        )
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_tensor = torch.from_numpy(image_rgb.transpose(2, 0, 1)).float() / 255.0
    return image_tensor, (original_height, original_width), scale


def build_extractor(spec: ModelSpec, device: torch.device):
    conf = {"name": spec.extractor_name, **spec.extractor_conf}
    model = dynamic_load(extractors, spec.extractor_name)
    return model(conf).eval().to(device)


def infer_aliked_model_name_from_weights(weight_path: Path) -> str:
    payload = torch.load(weight_path, map_location="cpu")
    state_dict = (
        payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    )
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Unexpected checkpoint format in {weight_path}")

    cleaned_state: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if isinstance(key, str):
            normalized_key = key[len("net.") :] if key.startswith("net.") else key
            cleaned_state[normalized_key] = value

    probe_key = "desc_head.agg_weights"
    if probe_key not in cleaned_state:
        raise RuntimeError(
            f"Could not infer ALIKED architecture: missing '{probe_key}' in {weight_path}"
        )

    channels = int(cleaned_state[probe_key].shape[0])
    if channels == 16:
        return "aliked-n16"
    if channels == 32:
        return "aliked-n32"
    raise RuntimeError(
        f"Unsupported ALIKED descriptor head channels ({channels}) in {weight_path}."
    )


def limit_features_by_budget(
    keypoints: np.ndarray,
    descriptors: np.ndarray,
    scores: np.ndarray,
    max_keypoints: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if max_keypoints <= 0 or keypoints.shape[0] <= max_keypoints:
        return keypoints, descriptors, scores

    if scores.shape[0] != keypoints.shape[0] or np.allclose(scores, scores[0]):
        selected = np.arange(max_keypoints, dtype=np.int64)
    else:
        selected = np.argpartition(scores, -max_keypoints)[-max_keypoints:]
        selected = selected[np.argsort(scores[selected])[::-1]]

    keypoints_limited = keypoints[selected]
    scores_limited = scores[selected]

    if descriptors.shape[0] == keypoints.shape[0]:
        descriptors_limited = descriptors[selected]
    elif descriptors.ndim > 1 and descriptors.shape[1] == keypoints.shape[0]:
        descriptors_limited = descriptors[:, selected]
    else:
        raise ValueError(
            f"Cannot slice descriptors of shape {descriptors.shape} with {keypoints.shape[0]} keypoints."
        )

    return keypoints_limited, descriptors_limited, scores_limited


def descriptors_to_nxd(descriptors: np.ndarray, num_keypoints: int) -> np.ndarray:
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
    outputs: Dict[str, Dict[str, np.ndarray]] = {}
    total_keypoints = 0

    for entry in tqdm(entries, desc=f"[{spec.label}] Features", leave=False):
        image_tensor, original_shape, scale = read_image_tensor(
            entry.image_path, grayscale=spec.grayscale
        )
        image_tensor = image_tensor.unsqueeze(0).to(device, non_blocking=True)
        prediction = extractor({"image": image_tensor})

        keypoints = prediction["keypoints"][0].detach().cpu().numpy().astype(np.float64)
        if scale < 1.0:
            keypoints = keypoints / scale

        descriptors = (
            prediction["descriptors"][0].detach().cpu().numpy().astype(np.float32)
        )
        score_tensor = prediction.get("scores", None)
        if score_tensor is None:
            scores = np.zeros(keypoints.shape[0], dtype=np.float32)
        else:
            scores = (
                score_tensor[0].detach().cpu().numpy().astype(np.float32).reshape(-1)
            )
        if scores.shape[0] != keypoints.shape[0]:
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

    average_keypoints = total_keypoints / max(1, len(entries))
    print(
        f"  -> Total keypoints kept: {total_keypoints} ({average_keypoints:.1f} per image)"
    )
    return outputs


def mutual_ratio_match(
    descriptors_left: np.ndarray,
    descriptors_right: np.ndarray,
    ratio: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if descriptors_left.shape[0] < 2 or descriptors_right.shape[0] < 2:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    matcher = cv2.BFMatcher(cv2.NORM_L2)

    knn_left_to_right = matcher.knnMatch(descriptors_left, descriptors_right, k=2)
    forward: Dict[int, int] = {}
    for pair in knn_left_to_right:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < ratio * second.distance:
            forward[first.queryIdx] = first.trainIdx

    knn_right_to_left = matcher.knnMatch(descriptors_right, descriptors_left, k=2)
    backward: Dict[int, int] = {}
    for pair in knn_right_to_left:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < ratio * second.distance:
            backward[first.queryIdx] = first.trainIdx

    idx_left: List[int] = []
    idx_right: List[int] = []
    for left_index, right_index in forward.items():
        if backward.get(right_index, -1) == left_index:
            idx_left.append(left_index)
            idx_right.append(right_index)

    if not idx_left:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    return np.asarray(idx_left, dtype=np.int64), np.asarray(idx_right, dtype=np.int64)


def match_features(
    feat_left: Dict[str, np.ndarray],
    feat_right: Dict[str, np.ndarray],
    ratio: float,
) -> Tuple[np.ndarray, np.ndarray]:
    keypoints_left = feat_left["keypoints"]
    keypoints_right = feat_right["keypoints"]
    if keypoints_left.shape[0] < 8 or keypoints_right.shape[0] < 8:
        return np.empty((0, 2)), np.empty((0, 2))

    descriptors_left = descriptors_to_nxd(
        feat_left["descriptors"], keypoints_left.shape[0]
    )
    descriptors_right = descriptors_to_nxd(
        feat_right["descriptors"], keypoints_right.shape[0]
    )
    idx_left, idx_right = mutual_ratio_match(
        descriptors_left, descriptors_right, ratio=ratio
    )
    if idx_left.size < 8:
        return np.empty((0, 2)), np.empty((0, 2))

    return keypoints_left[idx_left], keypoints_right[idx_right]


def normalize_keypoints(camera: pycolmap.Camera, keypoints: np.ndarray) -> np.ndarray:
    normalized = np.asarray(camera.cam_from_img(keypoints), dtype=np.float64)
    if normalized.ndim != 2 or normalized.shape[1] != 2:
        raise RuntimeError("camera.cam_from_img did not return an Nx2 array")
    return normalized


def ground_truth_relative_pose(
    left: ImageEntry, right: ImageEntry
) -> Tuple[np.ndarray, np.ndarray]:
    right_from_left = right.cam_from_world * left.cam_from_world.inverse()
    rotation_gt = np.asarray(right_from_left.rotation.matrix(), dtype=np.float64)
    translation_gt = np.asarray(right_from_left.translation, dtype=np.float64)
    return rotation_gt, translation_gt


def angle_error_mat(rotation_1: np.ndarray, rotation_2: np.ndarray) -> float:
    cos_angle = (np.trace(rotation_1.T @ rotation_2) - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.abs(np.arccos(cos_angle))))


def angle_error_vec(vector_1: np.ndarray, vector_2: np.ndarray) -> float:
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
    if points_left.shape[0] < 8 or points_right.shape[0] < 8:
        return float("inf")

    try:
        points_left_normalized = normalize_keypoints(left.camera, points_left)
        points_right_normalized = normalize_keypoints(right.camera, points_right)
    except Exception:
        return float("inf")

    valid = np.isfinite(points_left_normalized).all(axis=1) & np.isfinite(
        points_right_normalized
    ).all(axis=1)
    points_left = points_left[valid]
    points_right = points_right[valid]
    points_left_normalized = points_left_normalized[valid]
    points_right_normalized = points_right_normalized[valid]
    if points_left_normalized.shape[0] < 8:
        return float("inf")

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

    if fundamental.shape == (3, 3):
        fundamental_candidates = [fundamental]
    elif fundamental.shape[1] == 3 and fundamental.shape[0] % 3 == 0:
        fundamental_candidates = [
            fundamental[index : index + 3]
            for index in range(0, fundamental.shape[0], 3)
        ]
    else:
        return float("inf")

    inlier_mask = mask.astype(bool).reshape(-1)
    if inlier_mask.sum() < 5:
        return float("inf")

    calibration_left = np.asarray(left.camera.calibration_matrix(), dtype=np.float64)
    calibration_right = np.asarray(right.camera.calibration_matrix(), dtype=np.float64)

    best_rotation = None
    best_translation = None
    best_inlier_count = -1

    for fundamental_candidate in fundamental_candidates:
        essential = calibration_right.T @ fundamental_candidate @ calibration_left
        try:
            inlier_count, rotation_est, translation_est, _ = cv2.recoverPose(
                essential,
                points_left_normalized[inlier_mask],
                points_right_normalized[inlier_mask],
                cameraMatrix=np.eye(3),
            )
        except Exception:
            continue

        if inlier_count > best_inlier_count:
            best_inlier_count = int(inlier_count)
            best_rotation = rotation_est
            best_translation = translation_est.reshape(-1)

    if best_rotation is None or best_translation is None:
        return float("inf")

    rotation_gt, translation_gt = ground_truth_relative_pose(left, right)
    rotation_error = angle_error_mat(best_rotation, rotation_gt)
    translation_error = angle_error_vec(best_translation, translation_gt)
    if math.isfinite(translation_error):
        translation_error = min(translation_error, 180.0 - translation_error)

    total_error = max(rotation_error, translation_error)
    if not math.isfinite(total_error):
        return float("inf")
    return float(total_error)


def pose_auc(errors: Sequence[float], thresholds: Sequence[float]) -> List[float]:
    total_pairs = len(errors)
    values = np.asarray(errors, dtype=np.float64)
    values = values[np.isfinite(values)]

    if total_pairs == 0:
        return [0.0 for _ in thresholds]

    values = np.sort(values)
    recall = (np.arange(values.size, dtype=np.float64) + 1.0) / float(total_pairs)
    values = np.r_[0.0, values]
    recall = np.r_[0.0, recall]

    aucs: List[float] = []
    for threshold in thresholds:
        last_index = int(np.searchsorted(values, threshold, side="right"))
        if last_index <= 0:
            aucs.append(0.0)
            continue
        interpolated_recall = np.r_[recall[:last_index], recall[last_index - 1]]
        clipped_errors = np.r_[values[:last_index], threshold]
        aucs.append(
            float(np.trapezoid(interpolated_recall, x=clipped_errors) / threshold)
        )
    return aucs


def compute_recall_curve(
    errors: Sequence[float], max_threshold_deg: float = 20.0, num_points: int = 400
) -> Tuple[np.ndarray, np.ndarray]:
    total_pairs = len(errors)
    errors_array = np.asarray(errors, dtype=np.float64)
    finite = errors_array[np.isfinite(errors_array)]

    x_values = np.linspace(0.0, max_threshold_deg, num_points)
    if total_pairs == 0:
        return x_values, np.zeros_like(x_values)

    y_values = np.array(
        [np.sum(finite <= threshold) for threshold in x_values], dtype=np.float64
    ) / float(total_pairs)
    return x_values, y_values


def paper_color_cycle() -> List[str]:
    return [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#9467bd",
        "#ff7f0e",
        "#17becf",
    ]


def paper_style() -> None:
    palette = paper_color_cycle()
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "mathtext.fontset": "dejavuserif",
            "axes.prop_cycle": cycler(color=palette),
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
            "lines.linewidth": 2.2,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.7,
            "grid.linestyle": "-",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "legend.frameon": False,
        }
    )


def save_paper_plots(
    thresholds: Sequence[float],
    results: Dict[str, Dict[str, object]],
    output_dir: Path,
    prefix: str,
) -> None:
    paper_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = list(results.keys())
    colors = paper_color_cycle()
    color_by_label = {
        label: colors[index % len(colors)] for index, label in enumerate(labels)
    }

    max_threshold = float(max(thresholds)) if thresholds else 20.0
    curve_max = max(20.0, max_threshold)

    recall_path_pdf = output_dir / f"{prefix}_pose_recall_curve.pdf"
    recall_path_png = output_dir / f"{prefix}_pose_recall_curve.png"
    fig, ax = plt.subplots(figsize=(6.8, 4.1))
    for label in labels:
        x_values, y_values = compute_recall_curve(
            results[label]["errors"], max_threshold_deg=curve_max
        )
        ax.plot(x_values, y_values * 100.0, label=label, color=color_by_label[label])
    for threshold in thresholds:
        ax.axvline(
            float(threshold), color="#777777", linestyle="--", linewidth=0.9, alpha=0.55
        )
    ax.set_xlabel("Pose error threshold (degrees)")
    ax.set_ylabel("Pairs within threshold (%)")
    ax.set_xlim(0.0, curve_max)
    ax.set_ylim(0.0, 100.0)
    ax.set_title("Relative Pose Recall")
    ax.legend(loc="lower right", ncol=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(recall_path_pdf)
    fig.savefig(recall_path_png, dpi=300)
    plt.close(fig)

    auc_path_pdf = output_dir / f"{prefix}_auc_bars.pdf"
    auc_path_png = output_dir / f"{prefix}_auc_bars.png"
    x_positions = np.arange(len(thresholds), dtype=np.float64)
    width = 0.8 / max(1, len(labels))

    fig, ax = plt.subplots(figsize=(7.3, 4.4))
    for index, label in enumerate(labels):
        aucs = np.asarray(results[label]["aucs"], dtype=np.float64) * 100.0
        offset = (index - (len(labels) - 1) / 2.0) * width
        ax.bar(
            x_positions + offset,
            aucs,
            width=width,
            color=color_by_label[label],
            label=label,
        )

    tick_labels = [
        f"AUC@{int(value) if float(value).is_integer() else value}"
        for value in thresholds
    ]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel("AUC (%)")
    ax.set_ylim(0.0, 100.0)
    ax.set_title("Relative Pose AUC")
    ax.legend(loc="upper right", ncol=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(auc_path_pdf)
    fig.savefig(auc_path_png, dpi=300)
    plt.close(fig)

    summary_path = output_dir / f"{prefix}_plot_summary.txt"
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("Relative Pose Evaluation Plot Summary\n")
        handle.write("===================================\n")
        handle.write(
            f"Total pairs: {len(results[labels[0]]['errors']) if labels else 0}\n"
        )
        for label in labels:
            errors = np.asarray(results[label]["errors"], dtype=np.float64)
            successful = int(np.isfinite(errors).sum())
            handle.write(f"{label} successful pairs: {successful}/{len(errors)}\n")
        handle.write(f"Recall curve PDF: {recall_path_pdf}\n")
        handle.write(f"AUC bar PDF: {auc_path_pdf}\n")

    print(f"Saved plots to: {output_dir}")
    print(f"  - {recall_path_pdf.name}")
    print(f"  - {auc_path_pdf.name}")
    print(f"  - {summary_path.name}")


def make_all_pairs(names: Sequence[str]) -> List[Tuple[str, str]]:
    return list(itertools.combinations(names, 2))


def make_model_specs(
    selected_keys: Sequence[str], aliked_model_name: str, fine_weights: Path
) -> List[ModelSpec]:
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
    headers = [
        f"AUC@{int(value) if float(value).is_integer() else value}"
        for value in thresholds
    ]
    header_row = (
        "Model".ljust(22)
        + " | "
        + " | ".join(item.center(10) for item in headers)
        + " | Success"
    )
    separator = "-" * len(header_row)

    print("\n=== Relative Pose AUC Comparison ===")
    print(separator)
    print(header_row)
    print(separator)
    for model_label, model_result in results.items():
        aucs = model_result["aucs"]
        errors = model_result["errors"]
        success = int(np.isfinite(np.asarray(errors)).sum())
        values = [f"{float(score) * 100.0:8.2f}%" for score in aucs]
        row = (
            model_label.ljust(22)
            + " | "
            + " | ".join(values)
            + f" | {success}/{len(errors)}"
        )
        print(row)
    print(separator)


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
        left_entry = entries_by_name[left_name]
        right_entry = entries_by_name[right_name]
        feat_left = features_by_name[left_name]
        feat_right = features_by_name[right_name]

        points_left, points_right = match_features(
            feat_left, feat_right, ratio=fallback_ratio
        )
        error = estimate_pose_error_from_points(
            left=left_entry,
            right=right_entry,
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
        default=Path(
            "/zfs-pool/home/xbehoua00/filtered_images/200-400/2024_04_12_11_49_59/"
        ),
        help="Directory containing the images to evaluate.",
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=Path(
            "/zfs-pool/home/xbehoua00/zms-tool/xbehoua00/train/ALIKED/sift-rec-ref/sift/sfm_best/"
        ),
        help="COLMAP reconstruction directory containing cameras/images model files.",
    )
    parser.add_argument(
        "--fine-weights",
        type=Path,
        default=Path(
            "/zfs-pool/home/xbehoua00/recs/aliked_1080p_punisher_ep10_end.pth"
        ),
        help="Path to fine-tuned ALIKED .pth weights.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="auto",
        choices=["auto", "aliked-n16", "aliked-n32"],
        help="ALIKED architecture used by the base and fine-tuned ALIKED models.",
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
        help="Lowe ratio threshold for symmetric nearest-neighbor matching.",
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
    parser.add_argument(
        "--plot-prefix",
        type=str,
        default="evaluate_cov",
        help="File prefix for the generated plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds)
    selected_extractors = parse_extractor_list(args.extractors)

    if not args.image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {args.image_dir}")
    if not args.gt_dir.exists():
        raise FileNotFoundError(
            f"Ground-truth reconstruction directory not found: {args.gt_dir}"
        )
    if not args.fine_weights.exists():
        raise FileNotFoundError(f"Fine-tuned weights not found: {args.fine_weights}")
    if args.max_keypoints <= 0:
        raise ValueError(f"--max-keypoints must be > 0, got {args.max_keypoints}")

    model_name = args.model_name
    if model_name == "auto":
        model_name = infer_aliked_model_name_from_weights(args.fine_weights)
        print(f"Inferred ALIKED architecture from fine-tuned weights: {model_name}")

    model_specs = make_model_specs(selected_extractors, model_name, args.fine_weights)
    entries = load_entries(args.image_dir, args.gt_dir)
    if len(entries) < 2:
        raise RuntimeError(
            "Need at least 2 images shared between the image directory and GT model."
        )

    pairs = make_all_pairs([entry.name for entry in entries])
    print(f"Loaded {len(entries)} images with GT poses.")
    print(f"Evaluating {len(pairs)} image pairs.")
    print("Evaluating extractors: " + ", ".join(spec.key for spec in model_specs))
    print(
        "Protocol: IMC paper-style evaluation "
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
        aucs = pose_auc(errors, thresholds)
        results[spec.label] = {
            "key": spec.key,
            "errors": errors,
            "aucs": aucs,
        }

        del extractor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print_comparison_table(thresholds, results)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "thresholds": thresholds,
        "models": {
            model_label: {
                "key": result["key"],
                "aucs": result["aucs"],
                "total_pairs": len(result["errors"]),
                "successful_pairs": int(
                    np.isfinite(np.asarray(result["errors"])).sum()
                ),
            }
            for model_label, result in results.items()
        },
    }
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved metrics to: {args.output_json}")

    save_paper_plots(
        thresholds=thresholds,
        results=results,
        output_dir=args.output_json.parent,
        prefix=args.plot_prefix,
    )


if __name__ == "__main__":
    main()
