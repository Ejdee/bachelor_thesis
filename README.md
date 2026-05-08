# CA-ALIKED: Domain-Adapted Local Feature Extractor for Automotive Surfaces

This repository contains the implementation of a pipeline, which fine-tunes the [ALIKED](https://github.com/Shiaoming/ALIKED) feature extractor for 3D reconstruction of reflective vehicle surfaces without manual annotations. The approach uses a multi-model Structure-from-Motion consensus to automatically generate pseudo ground-truth keypoint heatmaps, then fine-tunes only the detection head of ALIKED while keeping the descriptor head frozen.

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
├── dataset/                       dataset download goes here (see Dataset section)
│   ├── train/                     unmasked training sequences
│   ├── imc_eval/                  pre-masked sequences for IMC pose evaluation
│   └── geo_eval/                  pre-masked sequences for RE / TL curve evaluation
├── preprocessing/
│   ├── run-preprocessing.py       ← single entry-point for the full preprocessing pipeline
│   ├── multi-model-sfm-pipeline.py multi-model SfM reconstruction
│   ├── pseudogt-scoring.py        pseudo-GT heatmap generation
│   ├── mask-rcnn-extractor.py     Mask R-CNN vehicle segmentation
│   └── apply-masks.py             applies masks to images
├── training/
│   ├── train-freeze-protocol.py   ← fine-tuning entry point
│   ├── pseudogt_dataset.py        dataset class for pseudo-GT heatmap training
│   ├── piecewise_semantic_loss.py custom loss for detection head fine-tuning
│   ├── aliked_frozen_wrapper.py   trainable ALIKED wrapper (frozen backbone)
│   └── clearml-hyperparameter-sweep.py ClearML hyperparameter sweep
├── evaluation/
│   ├── imc-pose-evaluation.py     IMC-protocol relative pose evaluation
│   ├── safe_pipeline_independent.py  ← SfM reconstruction entry point (all models or subset)
│   ├── reprojection-error-curves.py reprojection error threshold curves
│   ├── track-length-curves.py     track length threshold curves
│   └── run_eval_masked.sh         shell script for masked sequence evaluation
├── data_splits.txt                sequence IDs for train / IMC eval / geometric eval splits
└── dataset_splits.json            per-sequence frame selection used in experiments
```

---

## Dataset

The dataset is published on Zenodo: **https://doi.org/10.5281/zenodo.20082225**

Download and extract at the repository root:

```bash
unzip dataset.zip
```

After extraction the layout should be:

```
dataset/
├── train/
│   ├── 2024_04_09_16_34_06/    ← unmasked frames; input to preprocessing
│   └── ...
├── imc_eval/
│   ├── 2024_04_11_16_33_46/    ← pre-masked frames; input to IMC evaluation
│   └── ...
└── geo_eval/
    ├── 2024_07_09_16_48_13/    ← pre-masked frames; input to RE / TL evaluation
    └── ...
```

`imc_eval/` and `geo_eval/` images are already masked - no preprocessing needed for them. `train/` sequences must go through the preprocessing pipeline first to generate pseudo-GT heatmaps.

`data_splits.txt` lists the sequence IDs for each split. `dataset_splits.json` records the exact per-sequence frame selection used in the thesis experiments.

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
python -m pip install -e detectron2 --no-build-isolation  
```
or
```bash
git clone https://github.com/facebookresearch/detectron2.git
MAX_JOBS=1 python -m pip install -e detectron2 --no-build-isolation  
```

Make sure torch is already installed before this step. If the build fails with a GCC/nvcc version mismatch (common on Ubuntu with system-packaged CUDA), install gcc-11 and point the compiler to it:

```bash
sudo apt install gcc-11 g++-11
export CC=gcc-11 CXX=g++-11
export TORCH_CUDA_ARCH_LIST="7.5"   # adjust to your GPU's compute capability
python -m pip install -e detectron2 --no-build-isolation
```

**Step 3 - ALIKED**

```bash
git clone https://github.com/Shiaoming/ALIKED.git ALIKED
```

**Step 4 - everything else**

```bash
pip install -r requirements.txt
```

`hloc` is included as a local modified copy and does not need to be installed separately.

---

## 1. Preprocessing

Takes a folder of raw vehicle images and produces pseudo-GT heatmaps for training. Run once per training sequence:

```bash
python preprocessing/run-preprocessing.py \
    --images dataset/train/<sequence_id> \
    --output preprocessed/<sequence_id>
```

**What it does, in order:**

| Stage | Description |
|-------|-------------|
| 1/5 | Extract car masks with 5px dilation (used later for pseudo-GT filtering) |
| 2/5 | Extract car masks with 10px dilation (used for background removal) |
| 3/5 | Apply 10px masks to images (black out background) |
| 4/5 | Run multi-model SfM - DISK does full reconstruction, others triangulate against DISK geometry |
| 5/5 | Classify 3D points by quality (track length, reprojection error, cross-model agreement) and generate heatmaps |

**Output directory layout:**

```
output/
├── masks/
│   ├── dilate_5/          binary masks (5px)
│   └── dilate_10/         binary masks (10px)
├── masked/                background-removed images
├── reconstructions/       per-model SfM output (aliked/, disk/, sift/, superpoint/, r2d2/)
├── heatmaps/              pseudo-GT heatmaps (.npy) - input to training
└── valid_masks/           training region masks (.npy) - input to training
```

**Optional arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--gpu` | `0` | GPU device ID for mask extraction |
| `--score-thresh` | `0.5` | Quality score threshold for pseudo-GT point filtering |
| `--recon-attempts` | `10` | Max SfM retry attempts per model |

---

## 2. Training

Fine-tunes the ALIKED detection head on the generated pseudo-GT. Requires [ClearML](https://clear.ml/) for experiment tracking (free account).

```bash
python training/train-freeze-protocol.py \
    --frames-root       dataset/train \
    --preprocessed-root preprocessed/ \
    --checkpoint-dir    checkpoints/
```

`--frames-root` points to the unmasked training sequences; `--preprocessed-root` points to the directory produced by the preprocessing step. Both must contain matching per-sequence subdirectories:

```
dataset/train/
    2024_04_09_16_34_06/    ← original frames

preprocessed/
    2024_04_09_16_34_06/    ← output from run-preprocessing.py
        masked/
        heatmaps/
        valid_masks/
    ...
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

Both evaluation protocols require building SfM reconstructions first using `safe_pipeline_independent.py`. This script runs any subset of the seven baseline models on a single masked sequence:

| Model key | Method |
|-----------|--------|
| `sift` | SIFT + NN ratio |
| `disk` | DISK + LightGlue |
| `superpoint` | SuperPoint + LightGlue |
| `aliked` | ALIKED-n16 + LightGlue |
| `aliked_custom_lg` | CA-ALIKED (fine-tuned) + LightGlue |
| `r2d2` | R2D2 + NN mutual |
| `loftr` | LoFTR (dense) |

```bash
python evaluation/safe_pipeline_independent.py \
    --images          <path/to/masked/sequence> \
    --output          <output_dir> \
    --aliked_weights  checkpoints/aliked_ep10_end.pth \
    --models          <model_key> [<model_key> ...]   # omit to run all
```

Output per model:

```
<output_dir>/
    <model_key>/
        sfm_best/           ← best reconstruction (used by evaluation scripts)
        sfm-triangulation/  ← intermediate output
```

---

### Relative pose estimation (IMC protocol)

The SIFT ground truth and CA-ALIKED reconstructions are not included in the dataset and must be built first. Run on each `imc_eval` sequence:

```bash
# Build SIFT ground truth
python evaluation/safe_pipeline_independent.py \
    --images dataset/imc_eval/<sequence_id> \
    --output reconstructions/<sequence_id> \
    --models sift

# Build CA-ALIKED reconstruction
python evaluation/safe_pipeline_independent.py \
    --images         dataset/imc_eval/<sequence_id> \
    --output         reconstructions/<sequence_id> \
    --aliked_weights checkpoints/aliked_ep10_end.pth \
    --models         aliked_custom_lg
```

Then evaluate, pointing `--gt-dir` at the SIFT result:

```bash
python evaluation/imc-pose-evaluation.py \
    --image-dir    dataset/imc_eval/<sequence_id> \
    --gt-dir       reconstructions/<sequence_id>/sift/sfm_best \
    --fine-weights checkpoints/aliked_ep10_end.pth
```

Outputs AUC at 5°, 10°, and 20° thresholds, plus a bar chart and recall curve saved to the current directory.

---

### RE / TL threshold curves

All seven model reconstructions must be built for each `geo_eval` sequence. Run on each sequence:

```bash
python evaluation/safe_pipeline_independent.py \
    --images         dataset/geo_eval/<sequence_id> \
    --output         reconstructions/<sequence_id> \
    --aliked_weights checkpoints/aliked_ep10_end.pth
```

Then plot the curves across all five sequences:

```bash
# Reprojection error curves
python evaluation/reprojection-error-curves.py \
    --recon-root  reconstructions/ \
    --numbers     "2024_07_09_16_48_13 2024_06_30_14_18_10 2024_07_09_05_42_44 2024_07_09_16_43_11 2024_07_02_14_29_47" \
    --output-name figure_re.pdf

# Track length curves
python evaluation/track-length-curves.py \
    --recon-root  reconstructions/ \
    --numbers     "2024_07_09_16_48_13 2024_06_30_14_18_10 2024_07_09_05_42_44 2024_07_09_16_43_11 2024_07_02_14_29_47" \
    --output-name figure_tl.pdf
```
