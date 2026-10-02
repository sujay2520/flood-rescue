"""
src/train.py - PyTorch U-Net training pipeline on Sen1Floods11 SAR chips.

Requirements implemented:
  1. Sen1Floods11Dataset: Loads 256x256 SAR chips (VV or VV/VH bands) and binary flood masks.
     Supports random horizontal/vertical flips, 90-degree rotations, and SAR dB normalization.
     Includes automated fallback synthetic chip generation for zero-dependency local training verification.
  2. CombinedDiceBCELoss: Combines soft Dice Loss and Binary Cross Entropy with smooth factor.
  3. Model Architecture: Uses segmentation_models_pytorch.Unet with resnet18 or efficientnet-b0 encoder,
     pretrained on ImageNet (or randomly initialized if offline / download fails).
     1 or 2 input channels (SAR backscatter in dB), 1 output channel (flood probability).
  4. train_model: Complete training loop with gradient accumulation (e.g. accumulation_steps=4 for
     4GB GPU laptop / sm_61 architecture), validation IoU evaluation, checkpoint saving for best val_iou.
  5. CLI entrypoint with argparse: --data_dir, --epochs, --batch_size, --lr, --output, --in_channels,
     --encoder, --accumulation_steps, --device.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import random
import socket
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import rasterio
except ImportError:
    rasterio = None

try:
    import segmentation_models_pytorch as smp
except ImportError:
    smp = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train")


# ============================================================================
# Offline Connectivity Helper
# ============================================================================

def is_online(timeout: float = 1.0) -> bool:
    """Check whether host machine can reach external network."""
    if os.environ.get("FORCE_OFFLINE", "0") == "1":
        return False
    try:
        socket.setdefaulttimeout(timeout)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(("8.8.8.8", 53))
        return True
    except Exception:
        return False


# ============================================================================
# 1. Dataset & Augmentation Pipeline
# ============================================================================

def to_db(sar_array: np.ndarray) -> np.ndarray:
    """Convert linear SAR backscatter to decibels (dB). Clips to avoid log(0)."""
    clipped = np.clip(np.asarray(sar_array, dtype=np.float32), 1e-6, None)
    return 10.0 * np.log10(clipped)


def normalize_sar_band(
    band_data: np.ndarray,
    min_db: float = -30.0,
    max_db: float = 0.0,
) -> np.ndarray:
    """
    Normalize SAR backscatter in dB to [0.0, 1.0] range using min-max clipping.
    Typical Sentinel-1 SAR range:
      - VV: -30 dB (water / calm) to 0 dB (urban / vegetation)
      - VH: -35 dB to -5 dB
    """
    clipped = np.clip(band_data, min_db, max_db)
    norm = (clipped - min_db) / (max_db - min_db + 1e-7)
    return np.asarray(norm, dtype=np.float32)


class Sen1Floods11Dataset(Dataset):
    """
    PyTorch Dataset for Sentinel-1 SAR Flood Segmentation.

    Loads 256x256 SAR chips (VV or VV/VH bands) and binary flood mask targets.
    Supports random flips, 90-degree rotations, and backscatter normalization.
    If no chips exist in data_dir, generates realistic synthetic SAR/flood chip pairs.
    """

    def __init__(
        self,
        data_dir: str,
        in_channels: int = 2,
        chip_size: int = 256,
        is_train: bool = True,
        augment: bool = True,
        normalize: bool = True,
        num_synthetic_fallback: int = 24,
    ) -> None:
        self.data_dir = data_dir
        self.in_channels = in_channels
        self.chip_size = chip_size
        self.is_train = is_train
        self.augment = augment and is_train
        self.normalize = normalize
        self.num_synthetic_fallback = num_synthetic_fallback

        # Discover image and mask file pairs
        self.pairs: List[Tuple[str, str]] = self._find_pairs(data_dir)
        self.synthetic_mode = len(self.pairs) == 0

        if self.synthetic_mode:
            logger.warning(
                "No Sen1Floods11 chips found in '%s'. Enabled synthetic chip generator (%d chips).",
                data_dir,
                num_synthetic_fallback,
            )
            self.length = num_synthetic_fallback
        else:
            logger.info("Found %d paired Sen1Floods11 chips in '%s'.", len(self.pairs), data_dir)
            self.length = len(self.pairs)

    def _find_pairs(self, data_dir: str) -> List[Tuple[str, str]]:
        """Searches data_dir for S1 SAR images and matching ground-truth masks."""
        pairs: List[Tuple[str, str]] = []
        dpath = Path(data_dir)
        if not dpath.exists():
            return pairs

        # Candidate subfolders
        folder_combos = [
            ("S1Hand", "LabelHand"),
            ("v1.1/data/flood_events/HandLabeled/S1Hand", "v1.1/data/flood_events/HandLabeled/LabelHand"),
            ("images", "labels"),
            ("images", "masks"),
        ]

        for s1_sub, lbl_sub in folder_combos:
            s1_p = dpath / s1_sub
            lbl_p = dpath / lbl_sub
            if s1_p.is_dir() and lbl_p.is_dir():
                s1_files = sorted(glob.glob(str(s1_p / "*.tif*")) + glob.glob(str(s1_p / "*.png")))
                for f in s1_files:
                    name = Path(f).name
                    candidates = [
                        lbl_p / name.replace("S1Hand", "LabelHand"),
                        lbl_p / name.replace("_s1", "_label"),
                        lbl_p / name.replace("_sar", "_mask"),
                        lbl_p / name,
                    ]
                    for cand in candidates:
                        if cand.exists():
                            pairs.append((f, str(cand)))
                            break
                if pairs:
                    return pairs

        # Recursive search for *_S1Hand.tif and *_LabelHand.tif
        all_s1 = sorted(glob.glob(str(dpath / "**/*S1Hand*.tif*"), recursive=True))
        for s1_file in all_s1:
            lbl_file = s1_file.replace("S1Hand", "LabelHand")
            if os.path.exists(lbl_file):
                pairs.append((s1_file, lbl_file))
        if pairs:
            return pairs

        # Suffix matching in flat folder
        all_tifs = sorted(glob.glob(str(dpath / "*.tif*")))
        for f in all_tifs:
            if "_s1.tif" in f:
                cand = f.replace("_s1.tif", "_label.tif")
                if os.path.exists(cand):
                    pairs.append((f, cand))
            elif "_sar.tif" in f:
                cand = f.replace("_sar.tif", "_mask.tif")
                if os.path.exists(cand):
                    pairs.append((f, cand))

        return pairs

    def __len__(self) -> int:
        return self.length

    def _read_raster(self, path: str) -> np.ndarray:
        if rasterio is not None:
            with rasterio.open(path) as src:
                arr = src.read().astype(np.float32)  # Shape (C, H, W)
                return arr
        else:
            from PIL import Image
            img = Image.open(path)
            arr = np.array(img, dtype=np.float32)
            if arr.ndim == 2:
                return arr[np.newaxis, ...]
            elif arr.ndim == 3:
                return np.transpose(arr, (2, 0, 1))
            return arr

    def _generate_synthetic(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generates realistic synthetic 256x256 SAR backscatter chip and binary flood mask."""
        rng = np.random.RandomState(idx + (0 if self.is_train else 5000))
        h = w = self.chip_size

        # Dry land background backscatter (dB)
        vv = rng.normal(loc=-14.0, scale=2.8, size=(h, w)).astype(np.float32)
        vh = rng.normal(loc=-21.0, scale=3.0, size=(h, w)).astype(np.float32)

        mask = np.zeros((h, w), dtype=np.float32)

        # Inject synthetic flood inundation zone (calm water = dark SAR backscatter)
        if idx % 2 == 0 or idx % 3 == 0:
            cx = rng.randint(40, h - 40)
            cy = rng.randint(40, w - 40)
            radius = rng.randint(35, 75)
            y, x = np.ogrid[:h, :w]
            flood_region = ((x - cx) ** 2 + 1.8 * (y - cy) ** 2) <= (radius**2)
            vv[flood_region] = rng.normal(-24.0, 1.2, size=flood_region.sum())
            vh[flood_region] = rng.normal(-29.0, 1.3, size=flood_region.sum())
            mask[flood_region] = 1.0

        if self.in_channels == 1:
            sar = vv[np.newaxis, ...]
        else:
            sar = np.stack([vv, vh], axis=0)

        return sar, mask[np.newaxis, ...]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.synthetic_mode:
            sar, mask = self._generate_synthetic(idx)
        else:
            img_path, mask_path = self.pairs[idx]
            sar_raw = self._read_raster(img_path)
            mask_raw = self._read_raster(mask_path)

            # Select 1 (VV) or 2 (VV+VH) channels
            c_in = sar_raw.shape[0]
            if self.in_channels == 1:
                sar = sar_raw[0:1]
            else:
                if c_in >= 2:
                    sar = sar_raw[0:2]
                else:
                    sar = np.concatenate([sar_raw[0:1], sar_raw[0:1]], axis=0)

            # Convert to dB if values are in linear power scale
            valid = sar[np.isfinite(sar)]
            if valid.size > 0 and np.nanmean(valid) > 0 and np.nanmin(valid) >= 0:
                sar = to_db(sar)

            sar = np.nan_to_num(sar, nan=-30.0, posinf=0.0, neginf=-35.0)

            # Process ground truth mask (Sen1Floods11: 1=flood, 0=dry, -1=nodata)
            if mask_raw.ndim == 3:
                mask_raw = mask_raw[0]
            mask = (mask_raw == 1).astype(np.float32)[np.newaxis, ...]

        # Normalize SAR backscatter to [0, 1]
        if self.normalize:
            sar_norm = []
            for b_idx in range(sar.shape[0]):
                if b_idx == 0:
                    sar_norm.append(normalize_sar_band(sar[b_idx], min_db=-30.0, max_db=0.0))
                else:
                    sar_norm.append(normalize_sar_band(sar[b_idx], min_db=-35.0, max_db=-5.0))
            sar = np.stack(sar_norm, axis=0)

        # Ensure (C, chip_size, chip_size)
        _, h, w = sar.shape
        target = self.chip_size
        if h > target or w > target:
            if self.is_train:
                top = random.randint(0, h - target)
                left = random.randint(0, w - target)
            else:
                top = (h - target) // 2
                left = (w - target) // 2
            sar = sar[:, top : top + target, left : left + target]
            mask = mask[:, top : top + target, left : left + target]
        elif h < target or w < target:
            pad_h = max(0, target - h)
            pad_w = max(0, target - w)
            sar = np.pad(sar, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
            mask = np.pad(mask, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant", constant_values=0)

        # Synchronous data augmentations (flips & rotations)
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                sar = np.flip(sar, axis=-1)
                mask = np.flip(mask, axis=-1)
            # Random vertical flip
            if random.random() > 0.5:
                sar = np.flip(sar, axis=-2)
                mask = np.flip(mask, axis=-2)
            # Random 90-degree rotations
            k = random.choice([0, 1, 2, 3])
            if k > 0:
                sar = np.rot90(sar, k=k, axes=(-2, -1))
                mask = np.rot90(mask, k=k, axes=(-2, -1))

        sar_tensor = torch.from_numpy(np.ascontiguousarray(sar)).float()
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask)).float()
        return sar_tensor, mask_tensor


# ============================================================================
# 2. Combined Dice + BCE Loss & Metrics
# ============================================================================

class CombinedDiceBCELoss(nn.Module):
    """
    Combined Soft Dice Loss and Binary Cross Entropy with smooth factor.
    Compensates for severe class imbalance in flood segmentation tasks.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
        from_logits: bool = True,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.view_as(logits).float()

        if self.from_logits:
            bce_loss = F.binary_cross_entropy_with_logits(logits, targets)
            probs = torch.sigmoid(logits)
        else:
            probs = torch.clamp(logits, 1e-7, 1.0 - 1e-7)
            bce_loss = F.binary_cross_entropy(probs, targets)

        # Soft Dice calculation
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice.mean()

        return (self.bce_weight * bce_loss) + (self.dice_weight * dice_loss)


def compute_iou(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> float:
    """Computes Intersection-over-Union (IoU/Jaccard Index) for flood class."""
    with torch.no_grad():
        if (preds < 0).any() or (preds > 1).any():
            probs = torch.sigmoid(preds)
        else:
            probs = preds
        pred_bin = (probs >= threshold).float()
        target_bin = (targets >= 0.5).float()

        intersection = (pred_bin * target_bin).sum().item()
        union = pred_bin.sum().item() + target_bin.sum().item() - intersection
        return (intersection + eps) / (union + eps)


# ============================================================================
# 3. Model Architecture Setup (smp.Unet with resnet18 or efficientnet-b0)
# ============================================================================

def build_unet(
    encoder_name: str = "resnet18",
    in_channels: int = 2,
    classes: int = 1,
    pretrained: bool = True,
) -> nn.Module:
    """
    Builds smp.Unet with resnet18 or efficientnet-b0 encoder.
    Uses ImageNet pretrained weights if available; falls back to random
    initialization if working in offline / air-gapped environment.
    """
    if smp is None:
        raise ImportError(
            "segmentation-models-pytorch is required. Install via `pip install segmentation-models-pytorch`."
        )

    # If offline or offline requested, bypass network call immediately
    if not is_online():
        logger.info(
            "Offline environment detected. Initializing smp.Unet (%s) with random weights.",
            encoder_name,
        )
        return smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_channels,
            classes=classes,
            activation=None,
        )

    if pretrained:
        try:
            logger.info("Initializing smp.Unet (%s) with ImageNet weights...", encoder_name)
            model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights="imagenet",
                in_channels=in_channels,
                classes=classes,
                activation=None,
            )
            return model
        except Exception as exc:
            logger.warning(
                "Failed to load ImageNet weights (%s). Falling back to random initialization.", exc
            )

    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=in_channels,
        classes=classes,
        activation=None,
    )


# ============================================================================
# 4. Training Loop with Gradient Accumulation & Validation IoU
# ============================================================================

def train_model(
    data_dir: str = "data/sen1floods11",
    output_model_path: str = "models/best_unet_flood.pth",
    epochs: int = 15,
    batch_size: int = 4,
    lr: float = 1e-4,
    device: str = "cuda",
    in_channels: int = 2,
    encoder_name: str = "resnet18",
    accumulation_steps: int = 4,
    pretrained: bool = True,
    seed: int = 42,
) -> str:
    """
    Complete training loop with gradient accumulation, AMP, and validation IoU tracking.

    Args:
      data_dir: Directory containing Sen1Floods11 chips.
      output_model_path: Path to save best validation IoU model weights.
      epochs: Training epochs (default: 15).
      batch_size: Batch size (default: 4 for 4GB-8GB GPUs).
      lr: Learning rate (default: 1e-4).
      device: 'cuda' or 'cpu'. Automatically falls back to 'cpu' if CUDA not available.
      in_channels: 1 (VV) or 2 (VV+VH). Default: 2.
      encoder_name: 'resnet18' or 'efficientnet-b0'. Default: 'resnet18'.
      accumulation_steps: Gradient accumulation steps (default: 4).
      pretrained: Whether to use ImageNet weights (falls back to random if offline).
      seed: Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    target_device = torch.device(
        device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    )
    logger.info("Training on Device: %s (Requested: %s)", target_device, device)

    # 1. Dataset & Loaders
    train_dataset = Sen1Floods11Dataset(
        data_dir=data_dir,
        in_channels=in_channels,
        is_train=True,
        augment=True,
        normalize=True,
    )
    val_dataset = Sen1Floods11Dataset(
        data_dir=data_dir,
        in_channels=in_channels,
        is_train=False,
        augment=False,
        normalize=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=(len(train_dataset) > batch_size),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # 2. Model, Loss, Optimizer, Scaler
    model = build_unet(
        encoder_name=encoder_name,
        in_channels=in_channels,
        classes=1,
        pretrained=pretrained,
    ).to(target_device)

    criterion = CombinedDiceBCELoss(bce_weight=0.5, dice_weight=0.5, smooth=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    use_amp = (target_device.type == "cuda") and hasattr(torch.cuda.amp, "GradScaler")
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    os.makedirs(os.path.dirname(os.path.abspath(output_model_path)), exist_ok=True)

    best_val_iou = 0.0
    best_epoch = 0

    logger.info(
        "Beginning Training | Epochs: %d | Batch Size: %d | Accumulation Steps: %d | LR: %.1e",
        epochs,
        batch_size,
        accumulation_steps,
        lr,
    )

    for epoch in range(1, epochs + 1):
        # --- Training ---
        model.train()
        train_loss = 0.0
        train_iou = 0.0
        optimizer.zero_grad(set_to_none=True)

        for i, (imgs, masks) in enumerate(train_loader):
            imgs = imgs.to(target_device, non_blocking=True)
            masks = masks.to(target_device, non_blocking=True)

            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(imgs)
                    loss = criterion(logits, masks)
                    scaled_loss = loss / accumulation_steps
                scaler.scale(scaled_loss).backward()
            else:
                logits = model(imgs)
                loss = criterion(logits, masks)
                scaled_loss = loss / accumulation_steps
                scaled_loss.backward()

            # Step optimizer every accumulation_steps batches or at end of loader
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item()
            train_iou += compute_iou(logits, masks)

        avg_train_loss = train_loss / max(len(train_loader), 1)
        avg_train_iou = train_iou / max(len(train_loader), 1)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(target_device, non_blocking=True)
                masks = masks.to(target_device, non_blocking=True)

                if use_amp:
                    with torch.cuda.amp.autocast():
                        logits = model(imgs)
                        loss = criterion(logits, masks)
                else:
                    logits = model(imgs)
                    loss = criterion(logits, masks)

                val_loss += loss.item()
                val_iou += compute_iou(logits, masks)

        avg_val_loss = val_loss / max(len(val_loader), 1)
        avg_val_iou = val_iou / max(len(val_loader), 1)

        scheduler.step()

        logger.info(
            "Epoch [%02d/%02d] - Train Loss: %.4f, IoU: %.4f | Val Loss: %.4f, Val IoU: %.4f",
            epoch,
            epochs,
            avg_train_loss,
            avg_train_iou,
            avg_val_loss,
            avg_val_iou,
        )

        # Save Best Checkpoint
        if avg_val_iou >= best_val_iou:
            best_val_iou = avg_val_iou
            best_epoch = epoch
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_iou": float(best_val_iou),
                "encoder_name": encoder_name,
                "in_channels": in_channels,
            }
            torch.save(checkpoint, output_model_path)
            logger.info(" -> Saved new best model checkpoint to %s (Val IoU: %.4f)", output_model_path, best_val_iou)

    logger.info("Training Complete! Best Validation IoU: %.4f achieved at Epoch %d.", best_val_iou, best_epoch)
    return output_model_path


# ============================================================================
# 5. CLI Entrypoint
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train U-Net on Sentinel-1 SAR chips for Flood Inundation Mapping."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/sen1floods11",
        help="Path to folder containing Sen1Floods11 chips.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/best_unet_flood.pth",
        help="Target filepath to save best model checkpoint.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs (default: 15).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Mini-batch size (default: 4 for 4GB-8GB GPUs).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Initial learning rate (default: 1e-4).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Target device: 'cuda' or 'cpu'.",
    )
    parser.add_argument(
        "--in_channels",
        type=int,
        default=2,
        choices=[1, 2],
        help="Input SAR channels: 1 (VV) or 2 (VV+VH). Default: 2.",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="resnet18",
        choices=["resnet18", "efficientnet-b0"],
        help="Backbone encoder architecture (default: resnet18).",
    )
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4).",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        default=True,
        help="Attempt ImageNet pretrained encoder weights (default: True).",
    )
    parser.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false",
        help="Force randomly initialized encoder weights (offline mode).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_model(
        data_dir=args.data_dir,
        output_model_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        in_channels=args.in_channels,
        encoder_name=args.encoder,
        accumulation_steps=args.accumulation_steps,
        pretrained=args.pretrained,
    )


if __name__ == "__main__":
    main()
