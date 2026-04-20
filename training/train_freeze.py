import sys
import argparse
from pathlib import Path

# ALIKED source lives in final/ALIKED/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ALIKED"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as T
from tqdm import tqdm
import torch.nn.functional as F
import random
import cv2
import numpy as np
from nets.aliked import ALIKED
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from clearml import Task, Logger
from torch.optim.lr_scheduler import OneCycleLR

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


# ------------------------------------------------------------------
# ClearML experiment tracking
# ------------------------------------------------------------------
task = Task.init(project_name="ALIKED-Finetuning", task_name="CA-ALIKED")
task.connect(args)
logger = Logger.current_logger()


# ------------------------------------------------------------------
# Loss function
# ------------------------------------------------------------------
class CarFeaturePunisherLoss(nn.Module):
    def __init__(self, pos_weight=1.0, bg_weight=5.0, empty_car_weight=0.5):
        super().__init__()
        self.pos_weight = pos_weight
        self.bg_weight = bg_weight
        self.empty_car_weight = empty_car_weight

    def forward(self, pred, gt_hmap, train_mask, car_seg):
        # Verified keypoint region — L2 toward the Gaussian heatmap target
        pos_mask = (train_mask > 0.5).float() * (gt_hmap > 0.01).float()
        pos_loss = ((pred - gt_hmap) ** 2 * pos_mask).sum() / (pos_mask.sum() + 1e-6)

        # Background — L1 to drive responses to zero
        bg_mask = (car_seg < 0.5).float()
        bg_loss = (torch.abs(pred) * bg_mask).sum() / (bg_mask.sum() + 1e-6)

        # Unverified vehicle surface — soft L2 penalty (regularizer)
        empty_mask = (car_seg > 0.5).float() * (gt_hmap <= 0.01).float()
        empty_loss = (pred**2 * empty_mask).sum() / (empty_mask.sum() + 1e-6)

        return (
            self.pos_weight * pos_loss
            + self.bg_weight * bg_loss
            + self.empty_car_weight * empty_loss
        )


# ------------------------------------------------------------------
# Model — ALIKED with frozen backbone and descriptor head
# ------------------------------------------------------------------
class TrainableALIKED(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.net = ALIKED(
            model_name="aliked-n32", device=device, top_k=-1, scores_th=0.0, n_limit=0
        )

        for param in self.net.parameters():
            param.requires_grad = False
        for name, param in self.net.named_parameters():
            if "score_head" in name:
                param.requires_grad = True

        trainable = [n for n, p in self.net.named_parameters() if p.requires_grad]
        print(f"Trainable parameters ({len(trainable)}):")
        for n in trainable:
            print(f"  {n}")

    def forward(self, x):
        feature_map, score_map = self.net.extract_dense_map(x)
        desc_map = F.normalize(feature_map, p=2, dim=1)
        h, w = x.shape[2] // 8, x.shape[3] // 8
        desc_map = F.interpolate(
            desc_map, size=(h, w), mode="bilinear", align_corners=False
        )
        score_map = F.interpolate(
            score_map,
            size=(x.shape[2], x.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return score_map, desc_map


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------
class CarKeypointDataset(torch.utils.data.Dataset):
    def __init__(
        self, img_dir, masked_img_dir, label_dir, mask_dir, target_size=(1980, 1080)
    ):
        heatmap_files = sorted(Path(label_dir).glob("*.npy"))
        valid_stems = {f.stem for f in heatmap_files}
        all_imgs = sorted(
            list(Path(img_dir).glob("*.png")) + list(Path(img_dir).glob("*.jpg"))
        )
        self.imgs = [p for p in all_imgs if p.stem in valid_stems]
        print(f"  {len(heatmap_files)} heatmaps found → {len(self.imgs)} images loaded")

        self.masked_img_dir = Path(masked_img_dir)
        self.label_dir = Path(label_dir)
        self.mask_dir = Path(mask_dir)
        self.target_size = target_size

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        hmap_path = self.label_dir / f"{img_path.stem}.npy"
        mask_path = self.mask_dir / f"{img_path.stem}.npy"

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        masked_path = self.masked_img_dir / img_path.name
        if masked_path.exists():
            img_m = cv2.imread(str(masked_path), cv2.IMREAD_GRAYSCALE)
            car_seg = (img_m > 0).astype(np.float32)
        else:
            car_seg = np.ones(img.shape[:2], dtype=np.float32)

        hmap = np.load(hmap_path).astype(np.float32)
        train_mask = np.load(mask_path).astype(np.float32)

        W, H = self.target_size
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        car_seg = cv2.resize(car_seg, (W, H), interpolation=cv2.INTER_NEAREST)
        train_mask = cv2.resize(train_mask, (W, H), interpolation=cv2.INTER_NEAREST)
        hmap = cv2.resize(hmap, (W, H), interpolation=cv2.INTER_LINEAR)

        return (
            torch.from_numpy(img).permute(2, 0, 1),
            torch.from_numpy(hmap).unsqueeze(0),
            torch.from_numpy(train_mask).unsqueeze(0),
            torch.from_numpy(car_seg).unsqueeze(0),
        )


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
