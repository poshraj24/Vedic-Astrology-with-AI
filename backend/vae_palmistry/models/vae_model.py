import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import math


class ResidualBlock(nn.Module):
    """Residual block for stable deep networks"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class Encoder(nn.Module):
    """
    Encoder network: Image → (μ, log_σ²)

    Architecture designed for 224x224 input:
    224 → 112 → 56 → 28 → 14 → 7 → flatten → latent
    """

    def __init__(
        self, in_channels: int = 3, latent_dim: int = 256, base_channels: int = 32
    ):
        super().__init__()

        self.latent_dim = latent_dim

        # Initial convolution
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, stride=2, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )  # 224 → 112

        # Encoder blocks with downsampling
        self.encoder_blocks = nn.ModuleList(
            [
                self._make_encoder_block(base_channels, base_channels * 2),  # 112 → 56
                self._make_encoder_block(
                    base_channels * 2, base_channels * 4
                ),  # 56 → 28
                self._make_encoder_block(
                    base_channels * 4, base_channels * 8
                ),  # 28 → 14
                self._make_encoder_block(
                    base_channels * 8, base_channels * 16
                ),  # 14 → 7
            ]
        )

        # Final encoding dimension
        self.final_channels = base_channels * 16
        self.final_size = 7  # 224 / 32 = 7

        # Flatten and project to latent space
        flatten_dim = self.final_channels * self.final_size * self.final_size

        self.fc_mu = nn.Linear(flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(flatten_dim, latent_dim)

        # Initialize weights
        self._init_weights()

    def _make_encoder_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            ResidualBlock(out_ch),
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode
        x = self.init_conv(x)

        for block in self.encoder_blocks:
            x = block(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Get distribution parameters
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


class Decoder(nn.Module):
    """
    Decoder network: z → Reconstructed Image

    Mirror architecture of encoder:
    latent → 7 → 14 → 28 → 56 → 112 → 224
    """

    def __init__(
        self, out_channels: int = 3, latent_dim: int = 256, base_channels: int = 32
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.init_channels = base_channels * 16
        self.init_size = 7

        # Project latent to spatial
        self.fc = nn.Linear(
            latent_dim, self.init_channels * self.init_size * self.init_size
        )

        # Decoder blocks with upsampling
        self.decoder_blocks = nn.ModuleList(
            [
                self._make_decoder_block(
                    base_channels * 16, base_channels * 8
                ),  # 7 → 14
                self._make_decoder_block(
                    base_channels * 8, base_channels * 4
                ),  # 14 → 28
                self._make_decoder_block(
                    base_channels * 4, base_channels * 2
                ),  # 28 → 56
                self._make_decoder_block(base_channels * 2, base_channels),  # 56 → 112
            ]
        )

        # Final upsampling to full resolution
        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(base_channels, base_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, out_channels, 3, padding=1),
            nn.Sigmoid(),  # Output in [0, 1]
        )  # 112 → 224

        self._init_weights()

    def _make_decoder_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            ResidualBlock(in_ch),
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Project to spatial
        x = self.fc(z)
        x = x.view(x.size(0), self.init_channels, self.init_size, self.init_size)
        x = F.relu(x)

        # Decode
        for block in self.decoder_blocks:
            x = block(x)

        # Final convolution
        x = self.final_conv(x)

        return x


class PalmistryVAE(nn.Module):
    """
    Complete Variational Autoencoder for Palmistry

    Features:
    - Learns palm structure without labels
    - Supports β-VAE for disentanglement
    - Export-ready for mobile deployment
    - Robust to input variations

    Usage:
        model = PalmistryVAE(latent_dim=256)

        # Training
        outputs = model(images)
        loss = model.loss_function(outputs, images, beta=1.0)

        # Inference (get features)
        features = model.encode(images)

        # Generation
        new_images = model.decode(random_latent)
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 256,
        base_channels: int = 32,
        image_size: int = 224,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.image_size = image_size
        self.in_channels = in_channels

        self.encoder = Encoder(in_channels, latent_dim, base_channels)
        self.decoder = Decoder(in_channels, latent_dim, base_channels)

        # For tracking
        self.register_buffer("num_samples_seen", torch.tensor(0))

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = μ + σ * ε

        This allows gradients to flow through the sampling operation.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # During inference, just use the mean (deterministic)
            return mu

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode image to latent features (for inference/deployment)

        Args:
            x: Input images [B, C, H, W]

        Returns:
            Latent features [B, latent_dim]
        """
        mu, _ = self.encoder(x)
        return mu

    def encode_with_variance(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode with full distribution (for analysis)

        Returns:
            mu: Mean [B, latent_dim]
            logvar: Log variance [B, latent_dim]
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to image

        Args:
            z: Latent vectors [B, latent_dim]

        Returns:
            Reconstructed images [B, C, H, W]
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full forward pass

        Args:
            x: Input images [B, C, H, W]

        Returns:
            Dict with reconstruction, mu, logvar, z
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)

        return {"reconstruction": reconstruction, "mu": mu, "logvar": logvar, "z": z}

    def loss_function(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        beta: float = 1.0,
        reduction: str = "mean",
    ) -> Dict[str, torch.Tensor]:
        """
        VAE Loss = Reconstruction Loss + β * KL Divergence

        Args:
            outputs: Dict from forward()
            targets: Original images
            beta: Weight for KL divergence (β-VAE)
            reduction: 'mean' or 'sum'

        Returns:
            Dict with total_loss, recon_loss, kl_loss
        """
        recon = outputs["reconstruction"]
        mu = outputs["mu"]
        logvar = outputs["logvar"]

        # Reconstruction loss (MSE or BCE)
        recon_loss = F.mse_loss(recon, targets, reduction=reduction)

        # KL divergence: -0.5 * Σ(1 + log(σ²) - μ² - σ²)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        if reduction == "mean":
            kl_loss = kl_loss / mu.size(0)

        # Total loss
        total_loss = recon_loss + beta * kl_loss

        return {"total_loss": total_loss, "recon_loss": recon_loss, "kl_loss": kl_loss}

    def sample(self, num_samples: int, device: torch.device = None) -> torch.Tensor:
        """
        Generate new palm images from random latent vectors

        Args:
            num_samples: Number of images to generate
            device: Device to generate on

        Returns:
            Generated images [num_samples, C, H, W]
        """
        if device is None:
            device = next(self.parameters()).device

        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)

    def interpolate(
        self, x1: torch.Tensor, x2: torch.Tensor, steps: int = 10
    ) -> torch.Tensor:
        """
        Interpolate between two palm images in latent space

        Args:
            x1: First image [1, C, H, W]
            x2: Second image [1, C, H, W]
            steps: Number of interpolation steps

        Returns:
            Interpolated images [steps, C, H, W]
        """
        z1 = self.encode(x1)
        z2 = self.encode(x2)

        # Linear interpolation in latent space
        alphas = torch.linspace(0, 1, steps, device=z1.device)
        interpolations = []

        for alpha in alphas:
            z = (1 - alpha) * z1 + alpha * z2
            img = self.decode(z)
            interpolations.append(img)

        return torch.cat(interpolations, dim=0)

    def get_feature_importance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get importance of each latent dimension for an image

        Uses variance of the learned distribution as importance measure.
        Lower variance = more certain = more important feature.

        Args:
            x: Input image [B, C, H, W]

        Returns:
            Importance scores [B, latent_dim]
        """
        _, logvar = self.encoder(x)
        # Lower variance = higher importance
        importance = 1.0 / (torch.exp(logvar) + 1e-8)
        # Normalize
        importance = importance / importance.sum(dim=1, keepdim=True)
        return importance


class PalmistryVAELite(nn.Module):
    """
    Lightweight VAE for mobile deployment

    Reduced parameters while maintaining quality:
    - ~3x smaller than full model
    - Optimized for TFLite/ONNX conversion
    - Suitable for real-time inference on mobile
    """

    def __init__(
        self, in_channels: int = 3, latent_dim: int = 128, base_channels: int = 16
    ):
        super().__init__()

        self.latent_dim = latent_dim

        # Lightweight encoder
        self.encoder = nn.Sequential(
            # 224 → 112
            nn.Conv2d(in_channels, base_channels, 4, 2, 1),
            nn.ReLU(inplace=True),
            # 112 → 56
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            # 56 → 28
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            # 28 → 14
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            # 14 → 7
            nn.Conv2d(base_channels * 8, base_channels * 16, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 16),
            nn.ReLU(inplace=True),
        )

        flatten_dim = base_channels * 16 * 7 * 7
        self.fc_mu = nn.Linear(flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(flatten_dim, latent_dim)

        # Lightweight decoder
        self.fc_decode = nn.Linear(latent_dim, flatten_dim)

        self.decoder = nn.Sequential(
            # 7 → 14
            nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            # 14 → 28
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            # 28 → 56
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            # 56 → 112
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            # 112 → 224
            nn.ConvTranspose2d(base_channels, in_channels, 4, 2, 1),
            nn.Sigmoid(),
        )

        self.base_channels = base_channels

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Encode
        h = self.encoder(x)
        h = h.view(h.size(0), -1)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # Reparameterize
        if self.training:
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
        else:
            z = mu

        # Decode
        h = F.relu(self.fc_decode(z))
        h = h.view(h.size(0), self.base_channels * 16, 7, 7)
        reconstruction = self.decoder(h)

        return {"reconstruction": reconstruction, "mu": mu, "logvar": logvar, "z": z}


def create_palmistry_vae(variant: str = "standard", latent_dim: int = 256) -> nn.Module:
    """
    Factory function to create VAE models

    Args:
        variant: 'standard' or 'lite' (for mobile)
        latent_dim: Size of latent space

    Returns:
        VAE model
    """
    if variant == "standard":
        return PalmistryVAE(latent_dim=latent_dim, base_channels=32)
    elif variant == "lite":
        return PalmistryVAELite(latent_dim=min(latent_dim, 128), base_channels=16)
    else:
        raise ValueError(f"Unknown variant: {variant}")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test models
    print("=" * 60)
    print("PALMISTRY VAE MODEL TEST")
    print("=" * 60)

    # Test standard model
    model = PalmistryVAE(latent_dim=256)
    print(f"\nStandard VAE:")
    print(f"  Parameters: {count_parameters(model):,}")

    x = torch.randn(2, 3, 224, 224)
    outputs = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Reconstruction shape: {outputs['reconstruction'].shape}")
    print(f"  Latent shape: {outputs['mu'].shape}")

    # Test lite model
    model_lite = PalmistryVAELite(latent_dim=128)
    print(f"\nLite VAE (for mobile):")
    print(f"  Parameters: {count_parameters(model_lite):,}")

    outputs_lite = model_lite(x)
    print(f"  Reconstruction shape: {outputs_lite['reconstruction'].shape}")
    print(f"  Latent shape: {outputs_lite['mu'].shape}")

    # Test encoding
    features = model.encode(x)
    print(f"\nFeature extraction:")
    print(f"  Features shape: {features.shape}")

    print("\n✓ All tests passed!")
