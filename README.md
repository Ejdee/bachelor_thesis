# CA-ALIKED: Domain-Adapted Local Feature Extractor for Automotive Surfaces

This repository contains the implementation of the CA-ALIKED pipeline, which fine-tunes the [ALIKED](https://github.com/Shiaoming/ALIKED) feature extractor for 3D reconstruction of reflective vehicle surfaces without manual annotations. The approach uses a multi-model Structure-from-Motion consensus to automatically generate pseudo ground-truth keypoint heatmaps, then fine-tunes only the detection head of ALIKED while keeping the descriptor head frozen.

The code is organized into three stages that are meant to be run in sequence:

```
preprocessing/   →   training/   →   evaluation/
```

---

## Repository Structure

```
final/
├── ALIKED/                        ALIKED source (clone separately — see Dependencies)
├── hloc/                          Hierarchical Localization source (local, modified)
├── preprocessing/
│   ├── run_preprocessing.py       ← single entry-point for the full preprocessing pipeline
│   ├── pipeline_combined.py       multi-model SfM reconstruction
│   ├── point_classification.py    pseudo-GT heatmap generation
│   ├── car_mask_extractor.py      Mask R-CNN vehicle segmentation
│   └── apply_masks.py             applies masks to images
├── training/
│   ├── train_freeze.py            fine-tuning script (freeze protocol)
│   └── sweep_new.py               ClearML hyperparameter sweep
└── evaluation/
    ├── evaluate_cov.py            IMC-protocol relative pose evaluation
    ├── safe_pipeline_independent.py  single-model reconstruction for evaluation
    ├── re_threshold_curve_avg.py  reprojection error threshold curves
    └── tl_threshold_curve_avg.py  track length threshold curves
```

---

## Dependencies

**Step 1 - PyTorch with CUDA 12.1**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

CUDA 12.1 wheels are forward-compatible with CUDA 12.x drivers. Install this before everything else.

**Step 2 - detectron2**

detectron2 has no official wheel that reliably matches PyTorch + CUDA versions, so build it from source:

```bash
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
```

Make sure torch is already installed before this step. If the build fails with a GCC/nvcc version mismatch (common on Ubuntu with system-packaged CUDA), install gcc-11 and point the compiler to it:

```bash
sudo apt install gcc-11 g++-11
export CC=gcc-11 CXX=g++-11
export TORCH_CUDA_ARCH_LIST="7.5"   # adjust to your GPU's compute capability
python -m pip install -e detectron2
```

**Step 3 - ALIKED**

```bash
git clone https://github.com/Shiaoming/ALIKED.git ALIKED
```

Clone ALIKED into the `ALIKED/` directory at the repository root. The code is used as-is with no modifications.

**Step 4 - everything else**

```bash
pip install -r requirements.txt
```

`hloc` is included as a local modified copy and does not need to be installed separately.

---

## 1. Preprocessing

Takes a folder of raw vehicle images and produces pseudo-GT heatmaps for training.

```bash
python preprocessing/run_preprocessing.py \
    --images /path/to/sequence/frames \
    --output /path/to/output
```

**What it does, in order:**

| Stage | Description |
|-------|-------------|
| 1/5 | Extract car masks with 5px dilation (used later for pseudo-GT filtering) |
| 2/5 | Extract car masks with 10px dilation (used for background removal) |
| 3/5 | Apply 10px masks to images (black out background) |
| 4/5 | Run multi-model SfM — DISK does full reconstruction, others triangulate against DISK geometry |
| 5/5 | Classify 3D points by quality (track length, reprojection error, cross-model agreement) and generate heatmaps |

**Output directory layout:**

```
output/
├── masks/
│   ├── dilate_5/          binary masks (5px)
│   └── dilate_10/         binary masks (10px)
├── masked/                background-removed images
├── reconstructions/       per-model SfM output (aliked/, disk/, sift/, superpoint/, r2d2/)
├── heatmaps/              pseudo-GT heatmaps (.npy) — input to training
└── valid_masks/           training region masks (.npy) — input to training
```

**Optional arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--gpu` | `0` | GPU device ID for mask extraction |
| `--score-thresh` | `0.5` | Quality score threshold for pseudo-GT point filtering |
| `--recon-attempts` | `10` | Max SfM retry attempts per model |

Run `run_preprocessing.py` once per vehicle sequence. Repeat for as many sequences as you want to include in training.

---

## 2. Training

Fine-tunes the ALIKED detection head on the generated pseudo-GT. Requires [ClearML](https://clear.ml/) for experiment tracking (free account).

```bash
python training/train_freeze.py \
    --frames-root      /path/to/frames_root \
    --preprocessed-root /path/to/outputs_root \
    --checkpoint-dir   checkpoints/
```

`--frames-root` should contain per-sequence subdirectories with the original images.  
`--preprocessed-root` should contain the matching per-sequence subdirectories produced by preprocessing (each with `masked/`, `heatmaps/`, `valid_masks/`).

The script matches datasets by sequence folder name. In other words, if a sequence is named `2024_04_09_16_34_06`, both roots must contain that same subfolder.

Expected layout:
```
frames_root/
    2024_04_09_16_34_06/
        frame_00000.jpg
        ...
    2024_04_10_09_12_44/
        ...

outputs_root/
    2024_04_09_16_34_06/
        masked/
        heatmaps/
        valid_masks/
    2024_04_10_09_12_44/
        masked/
        heatmaps/
        valid_masks/
```

Important: do **not** pass a flat preprocessing folder that directly contains `heatmaps/`, `masked/`, and `valid_masks/` at its top level. Create one sequence subfolder first.

If you currently have a flat single-sequence output, restructure it like this:
```bash
SEQ=2024_04_09_16_34_06
mkdir -p /path/to/outputs_root/$SEQ
mv /path/to/flat_output/heatmaps /path/to/outputs_root/$SEQ/
mv /path/to/flat_output/valid_masks /path/to/outputs_root/$SEQ/
mv /path/to/flat_output/masked /path/to/outputs_root/$SEQ/
```

Then run training with:
```bash
python training/train_freeze.py \
    --frames-root /path/to/frames_root \
    --preprocessed-root /path/to/outputs_root \
    --checkpoint-dir checkpoints/
```

For example, if you ran preprocessing on two sequences:
```
frames_root/
    sequence_A/    ← original frames
    sequence_B/
outputs_root/
    sequence_A/    ← output from run_preprocessing.py
    sequence_B/
```

**Key training arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--lr` | `5e-5` | Peak learning rate |
| `--pos_weight` | `3.575` | Loss weight for verified keypoint regions |
| `--bg_weight` | `9.353` | Loss weight for background suppression |
| `--epochs` | `15` | Number of epochs |
| `--batch_size` | `9` | Batch size |
| `--use_all_gpus` | off | Wrap model in DataParallel |

Checkpoints are saved to `--checkpoint-dir/<clearml_task_id>/` after each epoch. The best-performing checkpoint (epoch 10 in the thesis) is named `aliked_ep10_end.pth`.

---

## 3. Evaluation

### Relative pose estimation (IMC protocol)

Evaluates keypoint-based pose accuracy on masked vehicle sequences against SIFT-based ground truth.

```bash
python evaluation/evaluate_cov.py \
    --image-dir /path/to/sequence/masked \
    --gt-dir    /path/to/sift_reference/sfm_best \
    --fine-weights /path/to/aliked_ep10_end.pth
```

Outputs AUC at 5°, 10°, and 20° thresholds, plus a bar chart and recall curve saved to the current directory.

### RE / TL threshold curves

Compares reprojection error and track length distributions across models for multiple reconstructions.

```bash
# Reprojection error curves
python evaluation/re_threshold_curve_avg.py \
    --recon-root /path/to/reconstructions \
    --numbers "seq1 seq2 seq3" \
    --output-name figure_re.pdf

# Track length curves
python evaluation/tl_threshold_curve_avg.py \
    --recon-root /path/to/reconstructions \
    --numbers "seq1 seq2 seq3" \
    --output-name figure_tl.pdf
```

`--recon-root` should be a directory where each subdirectory named by a sequence ID contains `aliked/sfm_best`, `disk/sfm_best`, etc.

### Single-model reconstruction (for evaluation sequences)

Runs the CA-ALIKED model through the full SfM pipeline to produce a reconstruction for evaluation.

```bash
python evaluation/safe_pipeline_independent.py \
    --images /path/to/masked/images \
    --output /path/to/output \
    --aliked_weights /path/to/aliked_ep10_end.pth
```

Recommended convention for compatibility with the RE/TL curve scripts:

Run this once per sequence, and set `--output` to `recon_root/<sequence_id>`.

Example:
```bash
python evaluation/safe_pipeline_independent.py \
    --images /data/masked/2024_04_09_16_34_06 \
    --output /data/recon_root/2024_04_09_16_34_06 \
    --aliked_weights checkpoints/aliked_ep10_end.pth
```

This produces model folders like:
```
/data/recon_root/2024_04_09_16_34_06/
    aliked_custom_lg/
        sfm_best/
        sfm-triangulation/
```

Then for curve plotting, use sequence IDs as `--numbers` and `recon_root` as `--recon-root`.
