import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pycolmap
from cycler import cycler

MODEL_ORDER = ["SIFT", "SuperPoint", "ALIKED", "DISK", "R2D2", "LoFTR", "CA-ALIKED"]

MODEL_COLORS = {
    "SIFT": "#1f77b4",
    "SuperPoint": "#ff7f0e",
    "ALIKED": "#9467bd",
    "DISK": "#2ca02c",
    "R2D2": "#d62728",
    "LoFTR": "#8c564b",
    "CA-ALIKED": "#ff00ff",
}


def paper_color_cycle():
    return [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#9467bd",
        "#ff7f0e",
        "#17becf",
        "#8c564b",
    ]


def paper_model_colors():
    palette = paper_color_cycle()
    return {
        "SIFT": palette[0],
        "SuperPoint": palette[4],
        "ALIKED": palette[3],
        "DISK": palette[2],
        "R2D2": palette[1],
        "LoFTR": palette[5],
        "CA-ALIKED": "#ff00ff",
    }


def paper_style() -> None:
    palette = paper_color_cycle()
    plt.rcParams.update(
        {
            "figure.figsize": (6.8, 4.1),  # Your original size
            # Use LaTeX rendering so it matches the document
            "text.usetex": False,
            "font.family": "serif",
            # --- MASSIVE FONT SIZES ---
            "font.size": 24,  # Base font size
            "axes.titlesize": 18,  # Plot title
            "axes.labelsize": 18,  # X and Y axis labels
            "xtick.labelsize": 18,  # X axis numbers
            "ytick.labelsize": 18,  # Y axis numbers
            "legend.fontsize": 18,  # Legend text
            # --- THICKER LINES ---
            # Thicker lines are required to balance the huge text
            "axes.linewidth": 1.5,  # Thicker box around the plot
            "lines.linewidth": 3.5,  # Thicker data lines
            # Standard aesthetics
            "axes.prop_cycle": cycler(color=palette),
            "axes.grid": True,
            "grid.alpha": 0.3,  # Darker grid so it doesn't vanish
            "grid.linewidth": 1.0,
            "grid.linestyle": "--",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "legend.frameon": False,
            # Export settings
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def parse_numbers(single_number: str, multiple_numbers_raw: str):
    if not multiple_numbers_raw.strip():
        return [single_number]
    numbers = [
        value.strip()
        for value in re.split(r"[\s,]+", multiple_numbers_raw.strip())
        if value.strip()
    ]
    if not numbers:
        return [single_number]
    return numbers


def build_models(number: str, recon_root: str):
    base = f"{recon_root}/{number}"
    return {
        "SIFT": f"{base}/sift/sfm_best",
        "SuperPoint": f"{base}/superpoint/sfm_best",
        "ALIKED": f"{base}/aliked/sfm_best",
        "DISK": f"{base}/disk/sfm_best",
        "R2D2": f"{base}/r2d2/sfm_best",
        "LoFTR": f"{base}/loftr/sfm_best",
        "CA-ALIKED": f"{base}/aliked_custom_lg/sfm_best",
    }


def get_model_data(model_path):
    try:
        reconstruction = pycolmap.Reconstruction(model_path)
        data = []
        for point3d in reconstruction.points3D.values():
            track_len = len(point3d.track.elements)
            error = point3d.error
            data.append((track_len, error))
        return data
    except Exception as exc:
        print(f"Failed to load model {model_path}: {exc}")
        return []


def get_sfm_candidates(model_path):
    normalized = os.path.normpath(model_path)
    parent_dir = os.path.dirname(normalized)
    if os.path.basename(parent_dir) == "models":
        sfm_best = os.path.dirname(parent_dir)
    else:
        sfm_best = normalized

    candidates = []
    models_dir = os.path.join(sfm_best, "models")
    model_children = []
    if os.path.isdir(models_dir):
        for child in os.listdir(models_dir):
            child_path = os.path.join(models_dir, child)
            if os.path.isdir(child_path):
                model_children.append(child_path)

        model_children.sort(
            key=lambda path: (
                0 if os.path.basename(path).isdigit() else 1,
                (
                    int(os.path.basename(path))
                    if os.path.basename(path).isdigit()
                    else os.path.basename(path)
                ),
            )
        )

    for candidate in [sfm_best, *model_children]:
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def load_best_model_data(model_path):
    candidates = get_sfm_candidates(model_path)
    best_path = None
    best_data = []

    for candidate in candidates:
        data = get_model_data(candidate)
        if len(data) > len(best_data):
            best_data = data
            best_path = candidate

    return best_path, best_data


def aggregate_re_threshold_curves(numbers, thresholds, recon_root):
    per_model_curves = {}

    for number in numbers:
        print(f"Processing reconstruction: {number}")
        models = build_models(number, recon_root)

        for model_name, path in models.items():
            best_path, data = load_best_model_data(path)
            if not data:
                continue

            errors = [err for _, err in data]
            counts_abs = np.asarray(
                [sum(err <= th for err in errors) for th in thresholds],
                dtype=np.float64,
            )

            if model_name not in per_model_curves:
                per_model_curves[model_name] = []
            per_model_curves[model_name].append(counts_abs)

            if best_path is not None:
                print(f"  {model_name}: {len(data)} points from {best_path}")

    aggregated = {}
    for model_name, curves in per_model_curves.items():
        stack = np.vstack(curves)
        aggregated[model_name] = {
            "mean": np.mean(stack, axis=0),
            "std": np.std(stack, axis=0),
            "num_runs": stack.shape[0],
        }

    return aggregated


def plot_re_threshold_curve_avg(thresholds, aggregated_curves, output_name):
    paper_style()
    model_colors = paper_model_colors()

    fig, ax = plt.subplots(figsize=(8.4, 5.0))

    model_sequence = [name for name in MODEL_ORDER if name in aggregated_curves]
    for name in aggregated_curves:
        if name not in model_sequence:
            model_sequence.append(name)

    for model_name in model_sequence:
        series = aggregated_curves[model_name]
        mean_curve = series["mean"]
        color = model_colors.get(model_name, MODEL_COLORS.get(model_name, "#333333"))

        (line,) = ax.plot(thresholds, mean_curve, label=model_name, color=color)

        if model_name == "CA-ALIKED":
            line.set_linewidth(3.5)  # optional emphasis

    ax.set_xlabel("Reprojection Error Threshold (px)", labelpad=8)
    ax.set_ylabel("Number of Points", labelpad=8)
    ax.set_title("Average RE Threshold Curve")
    ax.set_xlim(0.1, 2.0)
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend = ax.legend(loc="upper left", ncol=1, fontsize=14)

    for text in legend.get_texts():
        if text.get_text() == "CA-ALIKED":
            text.set_fontweight("bold")

    fig.tight_layout()
    fig.savefig(output_name)
    plt.close(fig)
    print(f"Saved {output_name}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot averaged RE threshold curves across multiple reconstructions."
    )
    parser.add_argument(
        "--recon-root",
        required=True,
        help="Base directory containing numbered reconstruction subdirs",
    )
    parser.add_argument(
        "--number",
        type=str,
        default="first",
        help="Single reconstruction ID (default: first)",
    )
    parser.add_argument(
        "--numbers",
        type=str,
        default="",
        help="Comma/space-separated reconstruction IDs for batch averaging",
    )
    parser.add_argument(
        "--output-name",
        default="figure2_re_curve_avg.pdf",
        help="Output filename (default: figure2_re_curve_avg.pdf)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    numbers = parse_numbers(args.number, args.numbers)
    thresholds = np.linspace(0.1, 2.0, 100)

    aggregated_curves = aggregate_re_threshold_curves(
        numbers, thresholds, args.recon_root
    )
    if not aggregated_curves:
        raise RuntimeError(
            "No valid model data found for the provided reconstructions."
        )

    plot_re_threshold_curve_avg(thresholds, aggregated_curves, args.output_name)
    print("Done.")


if __name__ == "__main__":
    main()
