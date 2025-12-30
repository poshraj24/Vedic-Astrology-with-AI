import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
from typing import List, Tuple, Optional, Callable
import random


class PalmDataset(Dataset):
    """
    Dataset for loading palm images

    Supports:
    - Multiple image formats (jpg, png, bmp, tiff)
    - On-the-fly augmentation
    - Consistent preprocessing for training/inference

    Usage:
        dataset = PalmDataset(
            root_dir='./tongji_dataset',
            transform=get_train_transforms()
        )
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        image_size: int = 224,
        recursive: bool = True,
    ):
        """
        Args:
            root_dir: Directory containing palm images
            transform: Optional transform to apply
            image_size: Target image size
            recursive: Search subdirectories for images
        """
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.transform = transform

        # Find all images
        self.image_paths = self._find_images(recursive)

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {root_dir}")

        print(f"Found {len(self.image_paths)} images in {root_dir}")

        # Default transform if none provided
        if self.transform is None:
            self.transform = self._default_transform()

    def _find_images(self, recursive: bool) -> List[Path]:
        """Find all image files in directory"""
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        images = []

        if recursive:
            for ext in extensions:
                images.extend(self.root_dir.rglob(f"*{ext}"))
                images.extend(self.root_dir.rglob(f"*{ext.upper()}"))
        else:
            for ext in extensions:
                images.extend(self.root_dir.glob(f"*{ext}"))
                images.extend(self.root_dir.glob(f"*{ext.upper()}"))

        return sorted(list(set(images)))

    def _default_transform(self) -> transforms.Compose:
        """Default transform for inference"""
        return transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img_path = self.image_paths[idx]

        try:
            # Load image
            image = Image.open(img_path).convert("RGB")

            # Apply transforms
            if self.transform:
                image = self.transform(image)

            return image

        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a random valid image instead
            new_idx = random.randint(0, len(self) - 1)
            if new_idx != idx:
                return self.__getitem__(new_idx)
            raise


class PalmAugmentation:
    """
    Palm-specific augmentation strategies

    Designed to be realistic for palm images:
    - Rotation (hands can be at slight angles)
    - Brightness/contrast (different lighting)
    - Color jitter (different skin tones, lighting)
    - Perspective (camera angles)
    """

    @staticmethod
    def get_train_transforms(image_size: int = 224) -> transforms.Compose:
        """Strong augmentation for training"""
        return transforms.Compose(
            [
                transforms.Resize((int(image_size * 1.1), int(image_size * 1.1))),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05
                ),
                transforms.RandomAffine(
                    degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)
                ),
                transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
                transforms.ToTensor(),
                # Normalize to [0, 1] for VAE (Sigmoid output)
            ]
        )

    @staticmethod
    def get_val_transforms(image_size: int = 224) -> transforms.Compose:
        """Minimal transforms for validation"""
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    @staticmethod
    def get_inference_transforms(image_size: int = 224) -> transforms.Compose:
        """
        Transforms for inference (matches mobile preprocessing)

        IMPORTANT: These must match exactly what you do on Android!
        """
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )


def create_dataloaders(
    train_dir: str,
    val_dir: Optional[str] = None,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    val_split: float = 0.1,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create training and validation dataloaders

    Args:
        train_dir: Directory with training images
        val_dir: Optional separate validation directory
        batch_size: Batch size
        image_size: Target image size
        num_workers: DataLoader workers
        val_split: Validation split ratio (if val_dir not provided)

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = PalmDataset(
        train_dir,
        transform=PalmAugmentation.get_train_transforms(image_size),
        image_size=image_size,
    )

    if val_dir:
        val_dataset = PalmDataset(
            val_dir,
            transform=PalmAugmentation.get_val_transforms(image_size),
            image_size=image_size,
        )
    else:
        # Split training data
        total_size = len(train_dataset)
        val_size = int(total_size * val_split)
        train_size = total_size - val_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

        # Apply validation transforms to val split
        # Note: This is a workaround since random_split doesn't change transforms
        val_dataset.dataset.transform = PalmAugmentation.get_val_transforms(image_size)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    return train_loader, val_loader


def preprocess_single_image(image_path: str, image_size: int = 224) -> torch.Tensor:
    """
    Preprocess a single image for inference

    Args:
        image_path: Path to image
        image_size: Target size

    Returns:
        Tensor ready for model [1, 3, H, W]
    """
    transform = PalmAugmentation.get_inference_transforms(image_size)

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image)

    # Add batch dimension
    return tensor.unsqueeze(0)


if __name__ == "__main__":
    print("Dataset module loaded successfully")
    print("\nUsage:")
    print("  from dataset import PalmDataset, create_dataloaders")
    print("  train_loader, val_loader = create_dataloaders('./palm_images')")
