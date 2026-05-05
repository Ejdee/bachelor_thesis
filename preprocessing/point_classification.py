from pathlib import Path
import pycolmap
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from collections import Counter
import cv2
import argparse
import sys
from tqdm import tqdm

# ==================== CONSTANTS ====================
RADIUS = 1.5  # Clustering radius multiplier
MASK_THRESHOLD = 127  # Threshold for binary mask
MIN_TRACK_LENGTH_RATIO = 0.05  # Minimum track length as fraction of views
MAX_REPROJECTION_ERROR = 5.0  # Maximum reprojection error for pre-filtering
MAX_GEOMETRY_ERROR = 2.0  # Maximum reprojection error for geometry check
MIN_GEOMETRY_TRACK_LENGTH_RATIO = 0.2  # Minimum track length for geometry
MIN_MODELS_FOR_IMAGE = 4  # Minimum number of models that must see an image
MASK_DILATION_RADIUS = 15  # Radius for dilating mask
# ====================================================

def robust_norm(x):
    lo, hi = np.percentile(x, [5, 95])
    print(f"  Robust norm percentiles: lo={lo:.4f}, hi={hi:.4f}")
    return np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)


def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def load_mask(img_name, masks_dir):
    mask_path = Path(masks_dir) / f"{Path(img_name).stem}_mask.png"
    if mask_path.exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            return (mask > MASK_THRESHOLD).astype(np.uint8)
    return None


def is_point_in_mask(uv, mask):
    if mask is None:
        return True
    u, v = map(int, uv)
    H, W = mask.shape
    return 0 <= u < W and 0 <= v < H and mask[v, u] > 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the original (unmasked) image frames",
    )
    parser.add_argument(
        "--masks",
        type=str,
        required=True,
        help="Path to the 5px-dilated mask directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the reconstructions directory (contains aliked/, disk/, …)",
    )
    parser.add_argument(
        "--agr",
        type=float,
        default=0.4,
        help="Weight for cross-model agreement in the S score (default: 0.4)",
    )
    parser.add_argument(
        "--tl",
        type=float,
        default=0.4,
        help="Weight for track length in the S score (default: 0.4)",
    )
    parser.add_argument(
        "--rerr",
        type=float,
        default=0.2,
        help="Weight for reprojection error in the S score (default: 0.2)",
    )
    parser.add_argument(
        "--score_thresh",
        type=float,
        default=0.33,
        help="Minimum S score threshold for accepting a point (default: 0.33)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Output directory for heatmaps, masks, and visualizations",
    )
    parser.add_argument(
        "--create_heatmaps",
        action="store_true",
        help="Generate training heatmaps and masks",
    )
    parser.add_argument(
        "--use_masks",
        action="store_true",
        help="Filter points by the car segmentation mask",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    masks_dir = Path(args.masks)
    recon_dir = Path(args.output)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load all five reconstructions
    # ------------------------------------------------------------------
    model_names = ["aliked", "disk", "sift", "superpoint", "r2d2"]
    model_paths = {
        "aliked": recon_dir / "aliked/sfm-triangulation",
        "disk": recon_dir / "disk/sfm-triangulation",
        "sift": recon_dir / "sift/sfm-triangulation",
        "superpoint": recon_dir / "superpoint/sfm-triangulation",
        "r2d2": recon_dir / "r2d2/sfm-triangulation",
    }
    recs = {
        name: pycolmap.Reconstruction(str(model_paths[name])) for name in model_names
    }
    NUM_OF_VIEWS = recs["disk"].num_cameras()

    # ------------------------------------------------------------------
    # Pre-filter: keep only points with TL >= 5% of views and RE <= 5.0
    # ------------------------------------------------------------------
    filtered_pts = []
    attr_all = []  # columns: [model_idx, local_id, track_len, reproj_err]

    for m_idx, name in enumerate(model_names):
        rec = recs[name]
        pts, tl, err = [], [], []
        for p in rec.points3D.values():
            pts.append(p.xyz)
            tl.append(p.track.length())
            err.append(float(getattr(p, "error", 0.0)))

        pts = np.array(pts, dtype=np.float64)
        tl = np.array(tl)
        err = np.array(err)

        keep = (tl >= NUM_OF_VIEWS * MIN_TRACK_LENGTH_RATIO) & (err <= MAX_REPROJECTION_ERROR)
        local_ids = np.nonzero(keep)[0]
        filtered_pts.append(pts[keep])
        attr_all.append(
            np.c_[np.full(keep.sum(), m_idx), local_ids, tl[keep], err[keep]]
        )

    pts_all = np.vstack(filtered_pts)
    attr_all = np.vstack(attr_all)
    print(f"After pre-filter: {pts_all.shape[0]} points")

    # ------------------------------------------------------------------
    # Spatial clustering via Union-Find (radius = RADIUS × median NN dist)
    # ------------------------------------------------------------------
    tree_all = cKDTree(pts_all)
    d2, _ = tree_all.query(pts_all, k=2)
    radius = RADIUS * np.median(d2[:, 1])
    print(f"Clustering radius: {radius:.4f}")

    parent = np.arange(len(pts_all))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, nbrs in enumerate(tree_all.query_ball_point(pts_all, radius)):
        for j in nbrs:
            if j != i:
                union(i, j)

    for i in range(len(pts_all)):
        parent[i] = find(i)

    _, inv = np.unique(parent, return_inverse=True)
    num_clusters = len(np.unique(parent))
    print(f"Number of clusters: {num_clusters}")

    # Cross-model agreement per cluster
    cluster_models = [set() for _ in range(num_clusters)]
    for i, c in enumerate(inv):
        cluster_models[c].add(int(attr_all[i, 0]))
    agree = np.array([len(s) for s in cluster_models], dtype=int)

    hist = np.bincount(agree, minlength=len(model_names) + 1)
    for i, count in enumerate(hist):
        print(
            f"  {count} clusters seen in {i} model(s) ({100 * count / num_clusters:.1f}%)"
        )

    # Attach agreement to each point
    attr_all = np.c_[
        attr_all, agree[inv]
    ]  # columns: [model_idx, local_id, tl, err, agree]

    tl_all = attr_all[:, 2]
    err_all = attr_all[:, 3]
    agr_all = attr_all[:, 4]

    # ------------------------------------------------------------------
    # Compute normalized quality score S and apply filters
    # ------------------------------------------------------------------
    T_n = robust_norm(tl_all)
    G_n = robust_norm(1.0 / (err_all + 1e-8))
    A_n = normalize(agr_all)

    S_all = args.tl * T_n + args.rerr * G_n + args.agr * A_n
    geom_ok = (err_all <= MAX_GEOMETRY_ERROR) & (tl_all >= NUM_OF_VIEWS * MIN_GEOMETRY_TRACK_LENGTH_RATIO)
    score_ok = S_all >= args.score_thresh
    keep = geom_ok | score_ok

    pts_all = pts_all[keep]
    attr_all = attr_all[keep]
    S_all = S_all[keep]
    geom_ok = geom_ok[keep]
    score_ok = score_ok[keep]

    # Pre-cache point ID lists for each model (needed for mask filtering below)
    pid_lists = {name: list(recs[name].points3D.keys()) for name in model_names}

    # ------------------------------------------------------------------
    # Optional mask-based filtering (filter 3)
    # ------------------------------------------------------------------
    if args.use_masks:
        print("Applying mask-based filtering...")
        mask_cache = {}
        mask_ok = np.ones(len(pts_all), dtype=bool)

        for i, p_attr in tqdm(
            enumerate(attr_all), total=len(attr_all), desc="Mask filter"
        ):
            model_idx = int(p_attr[0])
            local_id = int(p_attr[1])
            name = model_names[model_idx]
            rec = recs[name]

            pid_list = pid_lists[name]
            if local_id >= len(pid_list):
                mask_ok[i] = False
                continue
            pid = pid_list[local_id]
            if pid not in rec.points3D:
                mask_ok[i] = False
                continue

            p3 = rec.points3D[pid]
            visible = False
            for t in p3.track.elements:
                img_name = rec.images[t.image_id].name
                if img_name not in mask_cache:
                    mask_cache[img_name] = load_mask(img_name, masks_dir)
                mask = mask_cache[img_name]
                if mask is not None:
                    uv = rec.images[t.image_id].project_point(pts_all[i])
                    if uv is not None and is_point_in_mask(uv, mask):
                        visible = True
                        break

            if not visible:
                mask_ok[i] = False

        print(f"Mask filter: {mask_ok.sum()} / {len(mask_ok)} points remain")
        pts_all = pts_all[mask_ok]
        attr_all = attr_all[mask_ok]
        S_all = S_all[mask_ok]
        geom_ok = geom_ok[mask_ok]
        score_ok = score_ok[mask_ok]

    # Build reliable_pts list sorted by score
    reliable_pts = [
        {
            "idx": i,
            "point": pts_all[i],
            "score": float(S_all[i]),
            "tl": float(attr_all[i, 2]),
            "err": float(attr_all[i, 3]),
            "agr": float(attr_all[i, 4]),
            "model_idx": int(attr_all[i, 0]),
            "local_id": int(attr_all[i, 1]),
        }
        for i in range(len(pts_all))
    ]
    reliable_pts = sorted(reliable_pts, key=lambda p: p["score"])
    print(f"\nTotal reliable points: {len(reliable_pts)}")
    print("Top 10:")
    for pt in reliable_pts[-10:]:
        print(
            f"  S={pt['score']:.4f}  TL={pt['tl']:.1f}  ERR={pt['err']:.4f}  AGR={pt['agr']:.0f}"
        )

    if not args.create_heatmaps:
        sys.exit(0)

    # ------------------------------------------------------------------
    # Heatmap generation — only for images seen by >= 4 models
    # ------------------------------------------------------------------
    print("\nGenerating training heatmaps...")
    heatmap_dir = output_dir / "heatmaps"
    mask_dir = output_dir / "masks"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    image_sets = [{img.name for img in rec.images.values()} for rec in recs.values()]
    all_seen = [name for img_set in image_sets for name in img_set]
    image_counts = Counter(all_seen)
    common_images = {name for name, count in image_counts.items() if count >= MIN_MODELS_FOR_IMAGE}
    print(f"Images seen by 4+ models: {len(common_images)}")

    # Initialize heatmap and training mask arrays
    heatmaps = {}
    train_masks = {}
    sizes = {}

    for rec in recs.values():
        for image in rec.images.values():
            key = image.name
            if key not in common_images or key in heatmaps:
                continue
            img_temp = cv2.imread(str(dataset_dir / key), cv2.IMREAD_GRAYSCALE)
            if img_temp is None:
                continue
            h, w = img_temp.shape
            heatmaps[key] = np.zeros((h, w), dtype=np.float32)
            train_masks[key] = np.zeros((h, w), dtype=np.float32)
            sizes[key] = (h, w)

    # Project reliable points onto each image
    for p in tqdm(reliable_pts, desc="Projecting points"):
        name = model_names[int(p["model_idx"])]
        rec = recs[name]
        local_id = int(p["local_id"])
        pid_list = pid_lists[name]

        if local_id >= len(pid_list):
            continue
        pid = pid_list[local_id]
        if pid not in rec.points3D:
            continue
        p3 = rec.points3D[pid]

        for t in p3.track.elements:
            key = rec.images[t.image_id].name
            if key not in heatmaps:
                continue
            h, w = sizes[key]
            x, y = rec.images[t.image_id].points2D[t.point2D_idx].xy
            if 0 <= x < w and 0 <= y < h:
                heatmaps[key][int(round(y)), int(round(x))] = 1.0
                cv2.circle(train_masks[key], (int(x), int(y)), 15, 1.0, -1)

    # Smooth, normalize, and save
    model_colors = {
        "aliked": (0, 0, 255),
        "disk": (0, 255, 0),
        "sift": (255, 0, 0),
        "superpoint": (255, 255, 0),
        "r2d2": (255, 0, 255),
    }

    count = 0
    for key in tqdm(heatmaps, desc="Saving labels"):
        hmap = gaussian_filter(heatmaps[key], sigma=1.0)
        if hmap.max() > 1e-6:
            hmap /= hmap.max()
        mask = train_masks[key]

        np.save(heatmap_dir / f"{Path(key).stem}.npy", hmap)
        np.save(mask_dir / f"{Path(key).stem}.npy", mask)

        # Per-image debug overlay colored by contributing model
        img_debug = cv2.imread(str(dataset_dir / key))
        if img_debug is not None:
            h, w = img_debug.shape[:2]
            y_off = 30
            for m_name, color in model_colors.items():
                cv2.putText(
                    img_debug,
                    m_name,
                    (20, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                )
                y_off += 30

            for p in reliable_pts:
                name = model_names[int(p["model_idx"])]
                local_id = int(p["local_id"])
                pid_list = pid_lists[name]
                if local_id >= len(pid_list):
                    continue
                pid = pid_list[local_id]
                if pid not in recs[name].points3D:
                    continue
                p3 = recs[name].points3D[pid]
                if not any(
                    recs[name].images[t.image_id].name == key for t in p3.track.elements
                ):
                    continue
                t_match = next(
                    t
                    for t in p3.track.elements
                    if recs[name].images[t.image_id].name == key
                )
                uv = recs[name].images[t_match.image_id].project_point(p["point"])
                if uv is not None:
                    u, v = map(int, uv)
                    if 0 <= u < w and 0 <= v < h:
                        cv2.circle(
                            img_debug,
                            (u, v),
                            4,
                            model_colors.get(name, (255, 255, 255)),
                            -1,
                        )

            cv2.imwrite(str(heatmap_dir / f"{Path(key).stem}_points.png"), img_debug)

        count += 1

    print(f"\nGenerated {count} training pairs.")
    print("Feed the original RGB images to the network during training,")
    print("and use these heatmaps/masks for the loss computation.")


if __name__ == "__main__":
    main()
