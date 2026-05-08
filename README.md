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
│   ├── single-model-sfm.py        single-model reconstruction for evaluation
│   ├── reprojection-error-curves.py reprojection error threshold curves
│   ├── track-length-curves.py     track length threshold curves
│   └── run_eval_masked.sh         shell script for masked sequence evaluation
├── data_splits.txt                sequence IDs for train / IMC eval / geometric eval splits
└── dataset_splits.json            per-sequence frame selection used in experiments
```

---

## Dataset

The 3DRealCar dataset sequences used in this thesis are available on Google Drive:

https://drive.google.com/drive/folders/1csYZdO4sJmxV-tMQRPPVSiSp53_3y6fG

To download via command line:

```bash
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1csYZdO4sJmxV-tMQRPPVSiSp53_3y6fG
```

`data_splits.txt` lists the sequence IDs for each split (training, IMC evaluation, geometric evaluation). `dataset_splits.json` records the exact per-sequence frame selection used in the thesis experiments.

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

Takes a folder of raw vehicle images and produces pseudo-GT heatmaps for training.

```bash
python preprocessing/run-preprocessing.py \
    --images /path/to/sequence/frames \
    --output /path/to/output
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
├── heatmaps/              pseudo-GT heatmaps (.npy) — input to training
└── valid_masks/           training region masks (.npy) — input to training
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
    --frames-root       /path/to/frames_root \
    --preprocessed-root /path/to/outputs_root \
    --checkpoint-dir    checkpoints/
```

Both roots must contain matching per-sequence subdirectories:

```
frames_root/
    sequence_A/         ← original frames
    sequence_B/

outputs_root/
    sequence_A/         ← output from run-preprocessing.py
        masked/
        heatmaps/
        valid_masks/
    sequence_B/
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

### Relative pose estimation (IMC protocol)

Evaluates keypoint-based pose accuracy on masked vehicle sequences against SIFT-based ground truth.

```bash
python evaluation/imc-pose-evaluation.py \
    --image-dir    /path/to/sequence/masked \
    --gt-dir       /path/to/sift_reference/sfm_best \
    --fine-weights /path/to/aliked_ep10_end.pth
```

Outputs AUC at 5°, 10°, and 20° thresholds, plus a bar chart and recall curve saved to the current directory.

### RE / TL threshold curves

Compares reprojection error and track length distributions across models.

```bash
# Reprojection error curves
python evaluation/reprojection-error-curves.py \
    --recon-root  /path/to/reconstructions \
    --numbers     "seq1 seq2 seq3" \
    --output-name figure_re.pdf

# Track length curves
python evaluation/track-length-curves.py \
    --recon-root  /path/to/reconstructions \
    --numbers     "seq1 seq2 seq3" \
    --output-name figure_tl.pdf
```

`--recon-root` should be a directory where each subdirectory named by a sequence ID contains `aliked/sfm_best`, `disk/sfm_best`, etc.

### Single-model reconstruction

Runs CA-ALIKED through the full SfM pipeline to produce a reconstruction for evaluation.

```bash
python evaluation/single-model-sfm.py \
    --images         /path/to/masked/images \
    --output         /path/to/recon_root/<sequence_id> \
    --aliked_weights /path/to/aliked_ep10_end.pth
```

For compatibility with the RE/TL curve scripts, set `--output` to `recon_root/<sequence_id>`. This produces:

```
recon_root/sequence_id/
    aliked_custom_lg/
        sfm_best/
        sfm-triangulation/
```
