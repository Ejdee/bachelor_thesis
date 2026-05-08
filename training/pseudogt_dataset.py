"""=============================================================================
Project:     Accurate and Robust Localization of Landmarks on a Vehicle
Author:      Adam Běhoun <xbehoua00@vutbr.cz>
Year:        2026
Description: Dataset helpers for car keypoint training.
============================================================================="""

from pathlib import Path
import torch
import cv2
import numpy as np


class CarKeypointDataset(torch.utils.data.Dataset):
    """
    Class for loading car keypoints with heatmaps and segmentation masks.
    
    Loads original frames, masked frames (car segmentation), ground truth heatmaps,
    and training masks. Resizes all to a fixed target size.
    """
    def __init__(
        self, img_dir, masked_img_dir, label_dir, mask_dir, target_size=(1980, 1080)
    ):
        """Initializes the dataset by discovering valid image-label pairs and setting up paths.

        Args:
            img_dir: Directory containing original images.
            masked_img_dir: Directory containing masked frames (car segmentation).
            label_dir: Directory containing .npy heatmap files.
            mask_dir: Directory containing .npy training mask files.
            target_size (tuple, optional): Target size for resizing all inputs. Defaults to (1980, 1080).
        """
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

        # Load and normalize image
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Load car segmentation mask
        masked_path = self.masked_img_dir / img_path.name
        if masked_path.exists():
            img_m = cv2.imread(str(masked_path), cv2.IMREAD_GRAYSCALE)
            car_seg = (img_m > 0).astype(np.float32)
        else:
            car_seg = np.ones(img.shape[:2], dtype=np.float32)

        # Load heatmap and training mask
        hmap = np.load(hmap_path).astype(np.float32)
        train_mask = np.load(mask_path).astype(np.float32)

        # Resize all to target size
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
