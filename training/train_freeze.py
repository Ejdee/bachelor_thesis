import sys
import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as T
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from clearml import Task, Logger
from torch.optim.lr_scheduler import OneCycleLR

from models import TrainableALIKED
from losses import CarFeaturePunisherLoss
from datasets import CarKeypointDataset

parser = argparse.ArgumentParser(
    description="Fine-tune the ALIKED detection head on automotive data"
)
parser.add_argument(
    "--frames-root",
    type=str,
    required=True,
    help="Root directory containing per-sequence subdirs with original frames",
)
parser.add_argument(
    "--preprocessed-root",
    type=str,
    required=True,
    help="Root directory containing per-sequence subdirs with heatmaps/ and valid_masks/",
)
parser.add_argument(
    "--checkpoint-dir",
    type=str,
    default="checkpoints",
    help="Directory to save model checkpoints (default: checkpoints/)",
)
parser.add_argument(
    "--lr",
    type=float,
    default=5e-5,
    help="Peak learning rate for OneCycleLR (default: 5e-5)",
)
parser.add_argument(
    "--pos_weight",
    type=float,
    default=3.575,
    help="Loss weight for verified keypoint regions (default: 3.575)",
)
parser.add_argument(
    "--bg_weight",
    type=float,
    default=9.353,
    help="Loss weight for background suppression (default: 9.353)",
)
parser.add_argument(
    "--epochs", type=int, default=15, help="Number of training epochs (default: 15)"
)
parser.add_argument("--batch_size", type=int, default=9, help="Batch size (default: 9)")
parser.add_argument(
    "--use_all_gpus",
    action="store_true",
    help="Use all available GPUs via DataParallel",
)
args = parser.parse_args()


task = Task.init(project_name="ALIKED-Finetuning", task_name="CA-ALIKED")
task.connect(args)
logger = Logger.current_logger()


# ------------------------------------------------------------------
# Photometric augmentation
# ------------------------------------------------------------------
color_jitter = T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)


def apply_photometric_aug(img_tensor):
    img_tensor = color_jitter(img_tensor)
    if random.random() < 0.5:
        img_tensor = img_tensor + torch.randn_like(img_tensor) * 0.08
    return torch.clamp(img_tensor, 0.0, 1.0)


# ------------------------------------------------------------------
# ClearML heatmap visualization
# ------------------------------------------------------------------
def log_heatmap_visualization(logger, model, dataset, device, epoch):
    idx = random.randint(0, len(dataset) - 1)
    img, gt_hmap, train_mask, _ = dataset[idx]

    model.eval()
    with torch.no_grad():
        scores, _ = model(img.unsqueeze(0).to(device))
        pred = scores[0, 0].cpu().numpy()
    model.train()

    fig, axes = plt.subplots(1, 4, figsize=(12, 4))
    axes[0].imshow(img.permute(1, 2, 0).numpy())
    axes[0].set_title("Input")
    axes[0].axis("off")
    axes[1].imshow(gt_hmap[0].numpy(), cmap="jet")
    axes[1].set_title("GT")
    axes[1].axis("off")
    axes[2].imshow(train_mask[0].numpy(), cmap="gray")
    axes[2].set_title("Mask")
    axes[2].axis("off")
    axes[3].imshow(pred, cmap="jet")
    axes[3].set_title("Pred")
    axes[3].axis("off")
    plt.tight_layout()

    logger.report_matplotlib_figure(
        title="Qualitative", series="epoch", figure=fig, iteration=epoch
    )
    plt.close(fig)


# ------------------------------------------------------------------
# Device and model setup
# ------------------------------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"Using {torch.cuda.device_count()} GPU(s): {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Using CPU")

model = TrainableALIKED(device=device).to(device)
use_multi_gpu = torch.cuda.is_available() and (
    args.use_all_gpus or torch.cuda.device_count() > 1
)
if use_multi_gpu:
    model = nn.DataParallel(model)
    print("Model wrapped with DataParallel")


# ------------------------------------------------------------------
# Dataset discovery
# ------------------------------------------------------------------
frames_root = Path(args.frames_root)
preprocessed_root = Path(args.preprocessed_root)

if not preprocessed_root.exists():
    print(f"[ERROR] Preprocessed root not found: {preprocessed_root}")
    sys.exit(1)

datasets = []
for folder in sorted(d.name for d in preprocessed_root.iterdir() if d.is_dir()):
    frames_dir = frames_root / folder
    masked_dir = preprocessed_root / folder / "masked"
    heatmaps_dir = preprocessed_root / folder / "heatmaps"
    masks_dir = preprocessed_root / folder / "valid_masks"

    if not frames_dir.exists():
        print(f"[SKIP] {folder}: frames not found at {frames_dir}")
        continue
    missing = [
        n
        for n, p in [
            ("masked", masked_dir),
            ("heatmaps", heatmaps_dir),
            ("valid_masks", masks_dir),
        ]
        if not p.exists()
    ]
    if missing:
        print(f"[SKIP] {folder}: missing {', '.join(missing)}")
        continue

    print(f"Loading: {folder}")
    datasets.append(CarKeypointDataset(frames_dir, masked_dir, heatmaps_dir, masks_dir))

if not datasets:
    print("[ERROR] No valid datasets found.")
    sys.exit(1)

print(f"\nTotal datasets: {len(datasets)}")
loader = DataLoader(
    ConcatDataset(datasets), batch_size=args.batch_size, shuffle=True, num_workers=4
)


# ------------------------------------------------------------------
# Checkpoint directory
# ------------------------------------------------------------------
ckpt_dir = Path(args.checkpoint_dir) / task.id
ckpt_dir.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# Optimizer and loss
# ------------------------------------------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = OneCycleLR(
    optimizer,
    max_lr=args.lr,
    steps_per_epoch=len(loader),
    epochs=args.epochs,
    pct_start=0.1,
)
loss_fn = CarFeaturePunisherLoss(
    pos_weight=args.pos_weight, bg_weight=args.bg_weight
).to(device)

# Log initial zero values so ClearML doesn't error on NoneType
logger.report_scalar("reconstruction", "score", value=0, iteration=0)


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------
print(f"\nStarting training for {args.epochs} epoch(s)...")

for epoch in range(args.epochs):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
    midpoint = len(loader) // 2

    for batch_idx, (img, gt_hmap, train_mask, car_seg) in enumerate(pbar):
        img, gt_hmap = img.to(device), gt_hmap.to(device)
        train_mask = train_mask.to(device)
        car_seg = car_seg.to(device)

        img = apply_photometric_aug(img)

        scores, _ = model(img)
        loss = loss_fn(scores, gt_hmap, train_mask, car_seg)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        global_step = epoch * len(loader) + batch_idx
        logger.report_scalar("loss", "batch", loss.item(), global_step)

        # Save midpoint checkpoint
        if batch_idx == midpoint:
            state = (
                model.module.state_dict()
                if isinstance(model, nn.DataParallel)
                else model.state_dict()
            )
            ckpt = ckpt_dir / f"aliked_ep{epoch + 1}_mid.pth"
            torch.save(state, ckpt)
            print(f"  Midpoint checkpoint: {ckpt.name}")

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch + 1}: avg loss = {avg_loss:.6f}")
    logger.report_scalar("epoch", "avg_loss", avg_loss, epoch)

    state = (
        model.module.state_dict()
        if isinstance(model, nn.DataParallel)
        else model.state_dict()
    )
    torch.save(state, ckpt_dir / f"aliked_ep{epoch + 1}_end.pth")
    torch.save(state, ckpt_dir / "aliked_latest.pth")

    log_heatmap_visualization(logger, model, loader.dataset, device, epoch)

print(f"\nTraining complete. Checkpoints saved to: {ckpt_dir}")
