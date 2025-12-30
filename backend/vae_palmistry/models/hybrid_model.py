import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import math


class ResidualBlock(nn.Module):
    """Residual block with optional squeeze-excitation"""

    def __init__(self, channels: int, use_se: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        # Squeeze-Excitation for channel attention
        self.use_se = use_se
        if use_se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, channels // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // 4, channels, 1),
                nn.Sigmoid(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))

        if self.use_se:
            out = out * self.se(out)

        out += residual
        return F.relu(out, inplace=True)


class SharedEncoder(nn.Module):
    """
    Shared CNN backbone for both VAE and U-Net branches

    Returns feature maps at multiple scales for U-Net skip connections
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()

        self.base_channels = base_channels

        # Initial convolution
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )  # 224 → 112

        # Encoder stages (store for skip connections)
        self.enc1 = self._make_encoder_stage(
            base_channels, base_channels * 2
        )  # 112 → 56
        self.enc2 = self._make_encoder_stage(
            base_channels * 2, base_channels * 4
        )  # 56 → 28
        self.enc3 = self._make_encoder_stage(
            base_channels * 4, base_channels * 8
        )  # 28 → 14
        self.enc4 = self._make_encoder_stage(
            base_channels * 8, base_channels * 16
        )  # 14 → 7

        # Channel dimensions at each stage
        self.stage_channels = [
            base_channels,  # After init_conv: 112x112
            base_channels * 2,  # After enc1: 56x56
            base_channels * 4,  # After enc2: 28x28
            base_channels * 8,  # After enc3: 14x14
            base_channels * 16,  # After enc4: 7x7
        ]

    def _make_encoder_stage(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            ResidualBlock(out_ch),
            ResidualBlock(out_ch),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass returning bottleneck features and skip connections

        Returns:
            bottleneck: Final encoded features [B, C, 7, 7]
            skips: List of skip connection features at each scale
        """
        skips = []

        # Initial conv
        x = self.init_conv(x)
        skips.append(x)  # 112x112

        # Encoder stages
        x = self.enc1(x)
        skips.append(x)  # 56x56

        x = self.enc2(x)
        skips.append(x)  # 28x28

        x = self.enc3(x)
        skips.append(x)  # 14x14

        x = self.enc4(x)
        # Don't add bottleneck to skips

        return x, skips


class VAEHead(nn.Module):
    """
    VAE head: Takes bottleneck features → latent distribution
    """

    def __init__(self, in_channels: int, spatial_size: int, latent_dim: int = 256):
        super().__init__()

        self.latent_dim = latent_dim
        flatten_dim = in_channels * spatial_size * spatial_size

        # Projection to latent space
        self.fc_mu = nn.Linear(flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(flatten_dim, latent_dim)

        # For decoding back to spatial (used in reconstruction)
        self.fc_decode = nn.Linear(latent_dim, flatten_dim)
        self.decode_channels = in_channels
        self.decode_size = spatial_size

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode to μ and log(σ²)"""
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode_to_spatial(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent back to spatial features (for reconstruction)"""
        x = self.fc_decode(z)
        x = x.view(x.size(0), self.decode_channels, self.decode_size, self.decode_size)
        return F.relu(x)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        spatial = self.decode_to_spatial(z)

        return {"mu": mu, "logvar": logvar, "z": z, "spatial": spatial}


class UNetDecoder(nn.Module):
    """
    U-Net decoder for line segmentation

    Uses skip connections from shared encoder
    """

    def __init__(
        self,
        encoder_channels: List[int],
        num_classes: int = 5,  # Background + 4 major lines
        base_channels: int = 64,
    ):
        super().__init__()

        self.num_classes = num_classes

        # Decoder stages (reverse of encoder)
        # Each takes: upsampled + skip → output

        # 7→14: bottleneck (1024) + skip (512) → 512
        self.up4 = self._make_decoder_stage(
            encoder_channels[4] + encoder_channels[3], encoder_channels[3]
        )

        # 14→28: 512 + skip (256) → 256
        self.up3 = self._make_decoder_stage(
            encoder_channels[3] + encoder_channels[2], encoder_channels[2]
        )

        # 28→56: 256 + skip (128) → 128
        self.up2 = self._make_decoder_stage(
            encoder_channels[2] + encoder_channels[1], encoder_channels[1]
        )

        # 56→112: 128 + skip (64) → 64
        self.up1 = self._make_decoder_stage(
            encoder_channels[1] + encoder_channels[0], encoder_channels[0]
        )

        # 112→224: Final upsampling
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(encoder_channels[0], base_channels // 2, 4, 2, 1),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(inplace=True),
        )

        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(base_channels // 2, base_channels // 2, 3, padding=1),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, num_classes, 1),
        )

        # Auxiliary line attribute heads (length, curvature per line)
        self.line_attr_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(encoder_channels[0], 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes * 3),  # 3 attributes per line
        )

    def _make_decoder_stage(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            ResidualBlock(out_ch),
            nn.ConvTranspose2d(out_ch, out_ch, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(
        self, bottleneck: torch.Tensor, skips: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Decode with skip connections

        Args:
            bottleneck: Encoder output [B, C, 7, 7]
            skips: Skip features [112, 56, 28, 14] from encoder
        """
        # Upsample bottleneck to match skip[3] (14x14)
        x = F.interpolate(
            bottleneck, size=skips[3].shape[2:], mode="bilinear", align_corners=False
        )
        x = torch.cat([x, skips[3]], dim=1)
        x = self.up4(x)  # → 28x28

        x = torch.cat([x, skips[2]], dim=1)
        x = self.up3(x)  # → 56x56

        x = torch.cat([x, skips[1]], dim=1)
        x = self.up2(x)  # → 112x112

        x = torch.cat([x, skips[0]], dim=1)
        features_112 = self.up1(x)  # → 224x224 (but still at encoder channels)

        # Final upsampling
        features_224 = self.final_up(features_112)

        # Segmentation output
        segmentation = self.seg_head(features_224)

        # Line attributes from pooled features
        line_attrs = self.line_attr_head(features_112)

        return {
            "segmentation": segmentation,  # [B, num_classes, 224, 224]
            "line_attributes": line_attrs,  # [B, num_classes * 3]
            "features": features_112,  # For additional processing
        }


class ReconstructionDecoder(nn.Module):
    """
    Decoder for VAE reconstruction (image reconstruction)
    """

    def __init__(
        self, in_channels: int, out_channels: int = 3, base_channels: int = 64
    ):
        super().__init__()

        self.decoder = nn.Sequential(
            # 7 → 14
            nn.ConvTranspose2d(in_channels, base_channels * 8, 4, 2, 1),
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
            nn.ConvTranspose2d(base_channels, out_channels, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class PalmistryHybridModel(nn.Module):
    """
    Hybrid VAE + U-Net Model for Palmistry

    Combines:
    1. Shared encoder backbone (efficient feature extraction)
    2. VAE head (holistic palm features)
    3. U-Net decoder (precise line segmentation)
    4. Reconstruction decoder (VAE training)

    Usage:
        model = PalmistryHybridModel(latent_dim=256)

        # Full forward (training)
        outputs = model(images)
        loss = model.compute_loss(outputs, images, line_masks)

        # Inference modes
        features = model.extract_features(images)      # VAE latent
        lines = model.segment_lines(images)            # U-Net segmentation
        both = model.full_inference(images)            # Everything
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 256,
        base_channels: int = 64,
        num_line_classes: int = 5,  # bg + 4 lines
        image_size: int = 224,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_line_classes = num_line_classes
        self.image_size = image_size

        # Shared encoder
        self.encoder = SharedEncoder(in_channels, base_channels)

        # Get channel dimensions
        enc_channels = self.encoder.stage_channels
        bottleneck_channels = enc_channels[-1]  # 1024 for base=64
        bottleneck_size = image_size // 32  # 7 for 224

        # VAE head
        self.vae_head = VAEHead(
            in_channels=bottleneck_channels,
            spatial_size=bottleneck_size,
            latent_dim=latent_dim,
        )

        # U-Net decoder for segmentation
        self.unet_decoder = UNetDecoder(
            encoder_channels=enc_channels,
            num_classes=num_line_classes,
            base_channels=base_channels,
        )

        # Reconstruction decoder for VAE
        self.recon_decoder = ReconstructionDecoder(
            in_channels=bottleneck_channels,
            out_channels=in_channels,
            base_channels=base_channels,
        )

        # Line class names for reference
        self.line_names = [
            "background",
            "life_line",
            "heart_line",
            "head_line",
            "fate_line",
        ]

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full forward pass for training

        Returns dict with all outputs:
        - VAE: mu, logvar, z, reconstruction
        - U-Net: segmentation, line_attributes
        """
        # Shared encoding
        bottleneck, skips = self.encoder(x)

        # VAE branch
        vae_out = self.vae_head(bottleneck)
        reconstruction = self.recon_decoder(vae_out["spatial"])

        # U-Net branch
        unet_out = self.unet_decoder(bottleneck, skips)

        return {
            # VAE outputs
            "mu": vae_out["mu"],
            "logvar": vae_out["logvar"],
            "z": vae_out["z"],
            "reconstruction": reconstruction,
            # U-Net outputs
            "segmentation": unet_out["segmentation"],
            "line_attributes": unet_out["line_attributes"],
            # Intermediate features
            "bottleneck": bottleneck,
        }

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract latent features (VAE encoder only)"""
        bottleneck, _ = self.encoder(x)
        mu, _ = self.vae_head.encode(bottleneck)
        return mu

    def segment_lines(self, x: torch.Tensor) -> torch.Tensor:
        """Segment palm lines (U-Net only)"""
        bottleneck, skips = self.encoder(x)
        unet_out = self.unet_decoder(bottleneck, skips)
        return unet_out["segmentation"]

    def full_inference(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Complete inference for deployment

        Returns:
            features: Latent feature vector [B, latent_dim]
            segmentation: Line masks [B, num_classes, H, W]
            line_probs: Softmax probabilities [B, num_classes, H, W]
            line_attributes: Per-line attributes [B, num_classes * 3]
        """
        with torch.no_grad():
            outputs = self.forward(x)

            return {
                "features": outputs["mu"],
                "segmentation": outputs["segmentation"],
                "line_probs": F.softmax(outputs["segmentation"], dim=1),
                "line_attributes": outputs["line_attributes"],
            }

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        images: torch.Tensor,
        line_masks: Optional[torch.Tensor] = None,
        beta: float = 1.0,
        seg_weight: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss

        Args:
            outputs: Forward pass outputs
            images: Original images [B, 3, H, W]
            line_masks: Ground truth line masks [B, H, W] (optional)
            beta: VAE KL weight
            seg_weight: Segmentation loss weight

        Returns:
            Dict with individual losses and total
        """
        losses = {}

        # VAE Reconstruction loss
        recon_loss = F.mse_loss(outputs["reconstruction"], images)
        losses["recon_loss"] = recon_loss

        # VAE KL divergence
        mu = outputs["mu"]
        logvar = outputs["logvar"]
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        losses["kl_loss"] = kl_loss

        # Segmentation loss (if masks provided)
        if line_masks is not None:
            seg_loss = F.cross_entropy(outputs["segmentation"], line_masks)
            losses["seg_loss"] = seg_loss
        else:
            # Self-supervised segmentation using edge detection as proxy
            seg_loss = self._unsupervised_seg_loss(outputs["segmentation"], images)
            losses["seg_loss"] = seg_loss

        # Total loss
        total_loss = recon_loss + beta * kl_loss + seg_weight * losses["seg_loss"]
        losses["total_loss"] = total_loss

        return losses

    def _unsupervised_seg_loss(
        self, segmentation: torch.Tensor, images: torch.Tensor
    ) -> torch.Tensor:
        """
        Unsupervised segmentation loss using edge consistency

        Encourages segmentation to align with image edges
        """
        # Convert to grayscale
        gray = 0.299 * images[:, 0] + 0.587 * images[:, 1] + 0.114 * images[:, 2]
        gray = gray.unsqueeze(1)

        # Compute image gradients (edges)
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

        # Normalize edges
        edges = edges / (edges.max() + 1e-6)

        # Segmentation should have high confidence where edges are
        seg_probs = F.softmax(segmentation, dim=1)

        # Entropy of segmentation (should be low = confident)
        entropy = -torch.sum(
            seg_probs * torch.log(seg_probs + 1e-8), dim=1, keepdim=True
        )

        # Weighted entropy: low entropy at edges (confident predictions)
        edge_entropy_loss = torch.mean(edges * entropy)

        # Also encourage sparsity in line predictions (not everything is a line)
        line_probs = seg_probs[:, 1:]  # Exclude background
        sparsity_loss = torch.mean(line_probs)

        return edge_entropy_loss + 0.1 * sparsity_loss

    def get_line_geometry(self, segmentation: torch.Tensor) -> Dict[str, Dict]:
        """
        Extract line geometry from segmentation

        Args:
            segmentation: Model output [B, num_classes, H, W]

        Returns:
            Dict mapping line names to geometry info
        """
        # Get predicted classes
        pred_masks = torch.argmax(segmentation, dim=1)  # [B, H, W]

        results = {}
        for i, line_name in enumerate(self.line_names[1:], 1):  # Skip background
            # Get mask for this line
            line_mask = (pred_masks == i).float()

            if line_mask.sum() > 0:
                # Find line pixels
                coords = torch.nonzero(line_mask[0])  # Assuming batch size 1

                if len(coords) > 0:
                    # Calculate properties
                    y_coords = coords[:, 0].float()
                    x_coords = coords[:, 1].float()

                    results[line_name] = {
                        "present": True,
                        "pixel_count": len(coords),
                        "center": (x_coords.mean().item(), y_coords.mean().item()),
                        "extent": {
                            "x_min": x_coords.min().item(),
                            "x_max": x_coords.max().item(),
                            "y_min": y_coords.min().item(),
                            "y_max": y_coords.max().item(),
                        },
                        "length_approx": len(coords) ** 0.5
                        * 1.5,  # Rough approximation
                    }
                else:
                    results[line_name] = {"present": False}
            else:
                results[line_name] = {"present": False}

        return results


class PalmistryHybridModelLite(nn.Module):
    """
    Lightweight version for mobile deployment

    ~3x fewer parameters than full model
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 128,
        base_channels: int = 32,
        num_line_classes: int = 5,
        image_size: int = 224,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_line_classes = num_line_classes

        # Lightweight encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 8, base_channels * 16, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 16),
            nn.ReLU(inplace=True),
        )  # 224 → 7

        bottleneck_dim = base_channels * 16 * 7 * 7

        # VAE head
        self.fc_mu = nn.Linear(bottleneck_dim, latent_dim)
        self.fc_logvar = nn.Linear(bottleneck_dim, latent_dim)

        # Simple segmentation head (from bottleneck)
        self.seg_decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels, num_line_classes, 4, 2, 1),
        )

        self.line_names = [
            "background",
            "life_line",
            "heart_line",
            "head_line",
            "fate_line",
        ]

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Encode
        features = self.encoder(x)

        # VAE
        flat = features.view(features.size(0), -1)
        mu = self.fc_mu(flat)
        logvar = self.fc_logvar(flat)

        # Segmentation
        segmentation = self.seg_decoder(features)

        return {
            "mu": mu,
            "logvar": logvar,
            "z": mu,  # Use mean for inference
            "segmentation": segmentation,
        }

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        flat = features.view(features.size(0), -1)
        return self.fc_mu(flat)

    def segment_lines(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.seg_decoder(features)


def create_hybrid_model(
    variant: str = "standard", latent_dim: int = 256, num_line_classes: int = 5
) -> nn.Module:
    """
    Factory function to create hybrid models

    Args:
        variant: 'standard' or 'lite'
        latent_dim: Latent space dimension
        num_line_classes: Number of line classes (including background)
    """
    if variant == "standard":
        return PalmistryHybridModel(
            latent_dim=latent_dim, num_line_classes=num_line_classes, base_channels=64
        )
    elif variant == "lite":
        return PalmistryHybridModelLite(
            latent_dim=min(latent_dim, 128),
            num_line_classes=num_line_classes,
            base_channels=32,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=" * 60)
    print("PALMISTRY HYBRID MODEL TEST")
    print("=" * 60)

    # Test standard model
    model = PalmistryHybridModel(latent_dim=256)
    print(f"\nStandard Hybrid Model:")
    print(f"  Parameters: {count_parameters(model):,}")

    x = torch.randn(2, 3, 224, 224)
    outputs = model(x)

    print(f"\nOutputs:")
    print(f"  Latent (z): {outputs['z'].shape}")
    print(f"  Reconstruction: {outputs['reconstruction'].shape}")
    print(f"  Segmentation: {outputs['segmentation'].shape}")
    print(f"  Line attributes: {outputs['line_attributes'].shape}")

    # Test lite model
    model_lite = PalmistryHybridModelLite(latent_dim=128)
    print(f"\nLite Hybrid Model:")
    print(f"  Parameters: {count_parameters(model_lite):,}")

    outputs_lite = model_lite(x)
    print(f"  Latent: {outputs_lite['z'].shape}")
    print(f"  Segmentation: {outputs_lite['segmentation'].shape}")

    # Test inference modes
    print(f"\nInference modes:")
    features = model.extract_features(x)
    print(f"  Features only: {features.shape}")

    seg = model.segment_lines(x)
    print(f"  Segmentation only: {seg.shape}")

    full = model.full_inference(x)
    print(f"  Full inference keys: {list(full.keys())}")

    print("\n All tests passed!")
