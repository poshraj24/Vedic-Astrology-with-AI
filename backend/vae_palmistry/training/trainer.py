import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
import time
import json
from typing import Dict, Optional, Tuple
from datetime import datetime

try:
    from torch.utils.tensorboard import SummaryWriter

    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


class HybridModelTrainer:
    """
    Trainer for Palmistry Hybrid Model (VAE + U-Net)

    Handles:
    - Combined VAE + segmentation losses
    - Self-supervised line detection
    - Optional supervised segmentation
    - Loss balancing and scheduling
    """

    def __init__(
        self, model: nn.Module, config: Optional[Dict] = None, device: str = "auto"
    ):

        # Device setup
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        self.model = model.to(self.device)

        # Default config
        self.config = {
            # Training
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "epochs": 100,
            "patience": 15,
            # VAE settings
            "beta_start": 0.0,
            "beta_end": 0.1,  # Reduced from 1.0 for stability
            "beta_warmup_epochs": 20,  # Slower warmup
            # Loss weights
            "recon_weight": 1.0,
            "kl_weight": 0.001,  # Much smaller KL weight for stability
            "seg_weight": 0.5,  # Segmentation loss weight
            "seg_warmup_epochs": 5,  # Start seg loss after N epochs
            # Directories
            "checkpoint_dir": "./checkpoints",
            "log_dir": "./logs",
            "save_every": 10,
            # Training options
            "use_amp": torch.cuda.is_available(),
            "gradient_clip": 1.0,
        }

        if config:
            self.config.update(config)

        # Create directories
        Path(self.config["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
        Path(self.config["log_dir"]).mkdir(parents=True, exist_ok=True)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config["epochs"],
            eta_min=self.config["learning_rate"] * 0.01,
        )

        # Mixed precision
        self.scaler = GradScaler() if self.config["use_amp"] else None

        # Logging
        self.writer = None
        if HAS_TENSORBOARD:
            log_path = Path(self.config["log_dir"]) / datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
            self.writer = SummaryWriter(log_path)

        # State
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def _get_beta(self, epoch: int) -> float:
        """Get VAE beta for current epoch"""
        warmup = self.config["beta_warmup_epochs"]
        if epoch < warmup:
            return self.config["beta_start"] + (
                self.config["beta_end"] - self.config["beta_start"]
            ) * (epoch / warmup)
        return self.config["beta_end"]

    def _get_seg_weight(self, epoch: int) -> float:
        """Get segmentation loss weight (with warmup)"""
        warmup = self.config["seg_warmup_epochs"]
        if epoch < warmup:
            return 0.0  # No seg loss initially
        return self.config["seg_weight"]

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        images: torch.Tensor,
        line_masks: Optional[torch.Tensor],
        beta: float,
        seg_weight: float,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss

        Components:
        1. Reconstruction loss (VAE) - only if model has reconstruction
        2. KL divergence (VAE)
        3. Segmentation loss (U-Net) - supervised or self-supervised
        """
        losses = {}

        # 1. Reconstruction loss (only if model outputs it)
        if "reconstruction" in outputs:
            recon = outputs["reconstruction"]
            # Check for NaN in reconstruction
            if torch.isnan(recon).any():
                recon_loss = torch.tensor(0.0, device=images.device)
            else:
                recon_loss = F.mse_loss(recon, images)
            losses["recon"] = recon_loss
        else:
            # For lite model without reconstruction, use 0
            recon_loss = torch.tensor(0.0, device=images.device)
            losses["recon"] = recon_loss

        # 2. KL divergence with strong clamping
        mu = outputs["mu"]
        logvar = outputs["logvar"]

        # Clamp mu and logvar to prevent explosion
        mu = torch.clamp(mu, min=-100, max=100)
        logvar = torch.clamp(logvar, min=-20, max=20)

        # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # Clamp KL loss to reasonable range
        kl_loss = torch.clamp(kl_loss, min=0, max=1000)
        losses["kl"] = kl_loss

        # 3. Segmentation loss
        if seg_weight > 0:
            if line_masks is not None:
                # Supervised segmentation
                seg_loss = F.cross_entropy(outputs["segmentation"], line_masks)
            else:
                # Self-supervised using edge consistency
                seg_loss = self._compute_unsupervised_seg_loss(
                    outputs["segmentation"], images
                )
            losses["seg"] = seg_loss
        else:
            losses["seg"] = torch.tensor(0.0, device=images.device)

        # Combine losses - use smaller KL weight initially
        total = (
            self.config["recon_weight"] * recon_loss
            + beta * self.config["kl_weight"] * kl_loss
            + seg_weight * losses["seg"]
        )

        # Check for NaN and replace with 0
        if torch.isnan(total):
            total = recon_loss + losses["seg"]

        losses["total"] = total

        return losses

    def _compute_unsupervised_seg_loss(
        self, segmentation: torch.Tensor, images: torch.Tensor
    ) -> torch.Tensor:
        """
        Self-supervised segmentation loss

        Uses:
        1. Edge alignment - lines should align with image edges
        2. Sparsity - not everything is a line
        3. Continuity - lines should be connected
        """
        # Resize segmentation to match image size if needed
        if segmentation.shape[2:] != images.shape[2:]:
            segmentation = F.interpolate(
                segmentation,
                size=images.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        # Convert to grayscale
        gray = 0.299 * images[:, 0] + 0.587 * images[:, 1] + 0.114 * images[:, 2]
        gray = gray.unsqueeze(1)

        # Compute image gradients
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=images.dtype,
            device=images.device,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=images.dtype,
            device=images.device,
        ).view(1, 1, 3, 3)

        edge_x = F.conv2d(gray, sobel_x, padding=1)
        edge_y = F.conv2d(gray, sobel_y, padding=1)
        edges = torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)
        edges = edges / (edges.max() + 1e-6)

        # Segmentation probabilities
        seg_probs = F.softmax(segmentation, dim=1)
        line_probs = seg_probs[:, 1:]  # Exclude background

        # Loss 1: Lines should be at edges
        # High line probability at edge locations
        max_line_prob = line_probs.max(dim=1, keepdim=True)[0]
        edge_alignment = -torch.mean(edges * max_line_prob)

        # Loss 2: Sparsity - lines should be sparse
        sparsity = torch.mean(max_line_prob)

        # Loss 3: Entropy - predictions should be confident
        entropy = -torch.mean(torch.sum(seg_probs * torch.log(seg_probs + 1e-8), dim=1))

        # Combine
        loss = edge_alignment + 0.1 * sparsity + 0.05 * entropy

        return loss

    def train_epoch(
        self, train_loader: DataLoader, beta: float, seg_weight: float
    ) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()

        total_losses = {"total": 0, "recon": 0, "kl": 0, "seg": 0}
        num_batches = 0

        for batch in train_loader:
            # Handle both labeled and unlabeled data
            if isinstance(batch, dict):
                images = batch["image"].to(self.device)
                line_masks = batch.get("mask")
                if line_masks is not None:
                    line_masks = line_masks.to(self.device)
            else:
                images = batch.to(self.device)
                line_masks = None

            self.optimizer.zero_grad()

            # Forward pass
            if self.config["use_amp"] and self.scaler:
                with autocast():
                    outputs = self.model(images)
                    losses = self.compute_loss(
                        outputs, images, line_masks, beta, seg_weight
                    )

                self.scaler.scale(losses["total"]).backward()

                if self.config["gradient_clip"] > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config["gradient_clip"]
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                losses = self.compute_loss(
                    outputs, images, line_masks, beta, seg_weight
                )

                losses["total"].backward()

                if self.config["gradient_clip"] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config["gradient_clip"]
                    )

                self.optimizer.step()

            # Accumulate
            for key in total_losses:
                total_losses[key] += losses[key].item()
            num_batches += 1
            self.global_step += 1

        # Average
        return {k: v / num_batches for k, v in total_losses.items()}

    @torch.no_grad()
    def validate(
        self, val_loader: DataLoader, beta: float, seg_weight: float
    ) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()

        total_losses = {"total": 0, "recon": 0, "kl": 0, "seg": 0}
        num_batches = 0

        for batch in val_loader:
            if isinstance(batch, dict):
                images = batch["image"].to(self.device)
                line_masks = batch.get("mask")
                if line_masks is not None:
                    line_masks = line_masks.to(self.device)
            else:
                images = batch.to(self.device)
                line_masks = None

            outputs = self.model(images)
            losses = self.compute_loss(outputs, images, line_masks, beta, seg_weight)

            for key in total_losses:
                total_losses[key] += losses[key].item()
            num_batches += 1

        return {k: v / num_batches for k, v in total_losses.items()}

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: Optional[int] = None,
    ) -> Dict:
        """
        Full training loop

        Args:
            train_loader: Training data
            val_loader: Validation data (optional)
            epochs: Override config epochs

        Returns:
            Training history
        """
        epochs = epochs or self.config["epochs"]

        history = {
            "train_loss": [],
            "train_recon": [],
            "train_kl": [],
            "train_seg": [],
            "val_loss": [],
            "val_recon": [],
            "val_kl": [],
            "val_seg": [],
            "beta": [],
            "seg_weight": [],
            "lr": [],
        }

        print(f"\n{'='*60}")
        print("HYBRID MODEL TRAINING")
        print(f"{'='*60}")
        print(f"Epochs: {epochs}")
        print(f"Train batches: {len(train_loader)}")
        if val_loader:
            print(f"Val batches: {len(val_loader)}")
        print(f"{'='*60}\n")

        start_time = time.time()

        for epoch in range(epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Get current loss weights
            beta = self._get_beta(epoch)
            seg_weight = self._get_seg_weight(epoch)

            # Train
            train_losses = self.train_epoch(train_loader, beta, seg_weight)

            # Validate
            if val_loader:
                val_losses = self.validate(val_loader, beta, seg_weight)
            else:
                val_losses = {"total": 0, "recon": 0, "kl": 0, "seg": 0}

            # Update scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Record history
            history["train_loss"].append(train_losses["total"])
            history["train_recon"].append(train_losses["recon"])
            history["train_kl"].append(train_losses["kl"])
            history["train_seg"].append(train_losses["seg"])
            history["val_loss"].append(val_losses["total"])
            history["val_recon"].append(val_losses["recon"])
            history["val_kl"].append(val_losses["kl"])
            history["val_seg"].append(val_losses["seg"])
            history["beta"].append(beta)
            history["seg_weight"].append(seg_weight)
            history["lr"].append(current_lr)

            # TensorBoard logging
            if self.writer:
                self.writer.add_scalars("loss/train", train_losses, epoch)
                if val_loader:
                    self.writer.add_scalars("loss/val", val_losses, epoch)
                self.writer.add_scalar("params/beta", beta, epoch)
                self.writer.add_scalar("params/seg_weight", seg_weight, epoch)
                self.writer.add_scalar("params/lr", current_lr, epoch)

            # Print progress
            epoch_time = time.time() - epoch_start
            print(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"Train: {train_losses['total']:.4f} "
                f"(R:{train_losses['recon']:.4f} K:{train_losses['kl']:.4f} S:{train_losses['seg']:.4f}) | "
                f"Val: {val_losses['total']:.4f} | "
                f"β:{beta:.2f} seg:{seg_weight:.2f} | "
                f"{epoch_time:.1f}s"
            )

            # Checkpointing
            if (epoch + 1) % self.config["save_every"] == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")

            # Best model
            val_loss = val_losses["total"] if val_loader else train_losses["total"]
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best_model.pt")
                self.patience_counter = 0
                print(f"  → New best model!")
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= self.config["patience"]:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

        # Save final
        self.save_checkpoint("final_model.pt")

        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training completed in {total_time/60:.1f} minutes")
        print(f"Best loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}")

        # Save history
        history_path = Path(self.config["checkpoint_dir"]) / "history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        if self.writer:
            self.writer.close()

        return history

    def save_checkpoint(self, filename: str):
        """Save checkpoint"""
        path = Path(self.config["checkpoint_dir"]) / filename

        torch.save(
            {
                "epoch": self.current_epoch,
                "global_step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_val_loss": self.best_val_loss,
                "config": self.config,
            },
            path,
        )

    def load_checkpoint(self, filename: str):
        """Load checkpoint"""
        path = Path(self.config["checkpoint_dir"]) / filename
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]

        print(f"Loaded checkpoint from epoch {self.current_epoch}")

    @torch.no_grad()
    def visualize_outputs(
        self,
        val_loader: DataLoader,
        num_samples: int = 4,
        save_path: Optional[str] = None,
    ):
        """Visualize model outputs"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available")
            return

        self.model.eval()

        # Get batch
        batch = next(iter(val_loader))
        if isinstance(batch, dict):
            images = batch["image"][:num_samples].to(self.device)
        else:
            images = batch[:num_samples].to(self.device)

        outputs = self.model(images)

        # Plot
        fig, axes = plt.subplots(4, num_samples, figsize=(3 * num_samples, 12))

        for i in range(num_samples):
            # Original
            axes[0, i].imshow(images[i].cpu().permute(1, 2, 0).numpy())
            axes[0, i].set_title("Original")
            axes[0, i].axis("off")

            # Reconstruction
            recon = outputs["reconstruction"][i].cpu().permute(1, 2, 0).numpy()
            axes[1, i].imshow(recon)
            axes[1, i].set_title("Reconstruction")
            axes[1, i].axis("off")

            # Segmentation
            seg = outputs["segmentation"][i].cpu()
            seg_pred = torch.argmax(seg, dim=0).numpy()
            axes[2, i].imshow(seg_pred, cmap="tab10", vmin=0, vmax=4)
            axes[2, i].set_title("Segmentation")
            axes[2, i].axis("off")

            # Line probabilities (max over line classes)
            line_probs = F.softmax(seg, dim=0)[1:].max(dim=0)[0].numpy()
            axes[3, i].imshow(line_probs, cmap="hot")
            axes[3, i].set_title("Line Probability")
            axes[3, i].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved visualization to {save_path}")

        plt.close()


if __name__ == "__main__":
    print("Hybrid trainer module loaded")
    print("\nUsage:")
    print("  trainer = HybridModelTrainer(model)")
    print("  trainer.train(train_loader, val_loader)")
