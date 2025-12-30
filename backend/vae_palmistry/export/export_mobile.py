import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple, Dict
import json
import os


class PalmistryEncoder(nn.Module):
    """
    Wrapper that exports only the encoder part for inference

    This is what you deploy to Android - just feature extraction,
    no decoder needed for inference.
    """

    def __init__(self, vae_model: nn.Module):
        super().__init__()
        # Handle both VAE and Hybrid models
        if hasattr(vae_model, "encoder") and hasattr(vae_model, "vae_head"):
            # Hybrid model
            self.encoder = vae_model.encoder
            self.vae_head = vae_model.vae_head
            self.is_hybrid = True
        else:
            # Standard VAE
            self.encoder = vae_model.encoder
            self.is_hybrid = False

        self.latent_dim = vae_model.latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from palm image

        Args:
            x: Input image [B, 3, 224, 224] normalized to [0, 1]

        Returns:
            Feature vector [B, latent_dim]
        """
        if self.is_hybrid:
            bottleneck, _ = self.encoder(x)
            mu, _ = self.vae_head.encode(bottleneck)
            return mu
        else:
            mu, _ = self.encoder(x)
            return mu


class PalmistryHybridEncoder(nn.Module):
    """
    Export both features AND segmentation for hybrid model

    Returns:
    - features: Latent vector for holistic palm analysis
    - segmentation: Line mask predictions
    """

    def __init__(self, hybrid_model: nn.Module):
        super().__init__()
        self.encoder = hybrid_model.encoder
        self.vae_head = hybrid_model.vae_head
        self.unet_decoder = hybrid_model.unet_decoder
        self.latent_dim = hybrid_model.latent_dim
        self.num_line_classes = hybrid_model.num_line_classes

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features and segment lines

        Returns:
            features: [B, latent_dim]
            segmentation: [B, num_classes, H, W]
        """
        bottleneck, skips = self.encoder(x)

        # Features
        mu, _ = self.vae_head.encode(bottleneck)

        # Segmentation
        unet_out = self.unet_decoder(bottleneck, skips)

        return mu, unet_out["segmentation"]


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    image_size: int = 224,
    opset_version: int = 12,
    optimize: bool = True,
) -> str:
    """
    Export model to ONNX format

    Args:
        model: Trained VAE model
        output_path: Path to save ONNX model
        image_size: Input image size
        opset_version: ONNX opset version
        optimize: Apply ONNX optimizations

    Returns:
        Path to exported model
    """
    # Create encoder-only model
    encoder = PalmistryEncoder(model)
    encoder.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, image_size, image_size)

    # Export
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        encoder,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["features"],
        dynamic_axes={"input": {0: "batch_size"}, "features": {0: "batch_size"}},
    )

    print(f"Exported ONNX model to {output_path}")

    # Optimize if requested
    if optimize:
        try:
            import onnx
            from onnxruntime.transformers import optimizer

            optimized_path = output_path.with_suffix(".optimized.onnx")

            # Load and optimize
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)

            # Basic optimizations
            from onnx import optimizer as onnx_optimizer

            passes = [
                "eliminate_identity",
                "eliminate_nop_transpose",
                "fuse_consecutive_transposes",
                "fuse_bn_into_conv",
            ]
            optimized_model = onnx_optimizer.optimize(onnx_model, passes)

            onnx.save(optimized_model, str(optimized_path))
            print(f"Optimized ONNX model saved to {optimized_path}")

            return str(optimized_path)
        except ImportError:
            print("onnx/onnxruntime not installed, skipping optimization")

    return str(output_path)


def export_to_torchscript(
    model: nn.Module,
    output_path: str,
    image_size: int = 224,
    optimize_for_mobile: bool = True,
) -> str:
    """
    Export model to TorchScript format for PyTorch Mobile

    Args:
        model: Trained VAE model
        output_path: Path to save TorchScript model
        image_size: Input image size
        optimize_for_mobile: Apply mobile optimizations

    Returns:
        Path to exported model
    """
    # Create encoder-only model
    encoder = PalmistryEncoder(model)
    encoder.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, image_size, image_size)

    # Trace the model
    traced_model = torch.jit.trace(encoder, dummy_input)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if optimize_for_mobile:
        try:
            from torch.utils.mobile_optimizer import optimize_for_mobile

            traced_model = optimize_for_mobile(traced_model)
            print("Applied mobile optimizations")
        except ImportError:
            print("Mobile optimizer not available, saving unoptimized")

    # Save
    traced_model.save(str(output_path))
    print(f"Exported TorchScript model to {output_path}")

    return str(output_path)


def export_to_tflite(
    model: nn.Module, output_path: str, image_size: int = 224, quantize: bool = False
) -> Optional[str]:
    """
    Export model to TensorFlow Lite format

    Requires: onnx, onnx-tf, tensorflow

    Args:
        model: Trained VAE model
        output_path: Path to save TFLite model
        image_size: Input image size
        quantize: Apply int8 quantization

    Returns:
        Path to exported model or None if failed
    """
    try:
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf
    except ImportError as e:
        print(f"TFLite export requires additional packages: {e}")
        print("Install with: pip install onnx onnx-tf tensorflow")
        return None

    # First export to ONNX
    onnx_path = Path(output_path).with_suffix(".onnx")
    export_to_onnx(model, str(onnx_path), image_size, optimize=False)

    # Load ONNX model
    onnx_model = onnx.load(str(onnx_path))

    # Convert to TensorFlow
    tf_rep = prepare(onnx_model)

    # Save as SavedModel
    saved_model_path = Path(output_path).with_suffix("")
    tf_rep.export_graph(str(saved_model_path))

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_path))

    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()

    # Save TFLite model
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    print(f"Exported TFLite model to {output_path}")

    # Cleanup intermediate files
    os.remove(onnx_path)
    import shutil

    shutil.rmtree(saved_model_path, ignore_errors=True)

    return str(output_path)


def create_android_config(
    output_dir: str, model_name: str, latent_dim: int, image_size: int = 224
) -> str:
    """
    Create configuration file for Android app

    This JSON file tells your Android app how to:
    - Preprocess images
    - Interpret model outputs
    - Handle the feature vector

    Args:
        output_dir: Directory to save config
        model_name: Name of the model file
        latent_dim: Size of latent space
        image_size: Input image size

    Returns:
        Path to config file
    """
    config = {
        "model_info": {
            "name": "PalmistryVAE",
            "version": "1.0.0",
            "model_file": model_name,
            "latent_dim": latent_dim,
        },
        "preprocessing": {
            "input_size": [image_size, image_size],
            "input_channels": 3,
            "normalize": {
                "mean": [0.0, 0.0, 0.0],  # No mean subtraction (images in [0,1])
                "std": [1.0, 1.0, 1.0],
            },
            "pixel_range": [0.0, 1.0],
            "channel_order": "RGB",
        },
        "output": {
            "feature_dim": latent_dim,
            "feature_names": [f"latent_{i}" for i in range(latent_dim)],
        },
        "android_code_hint": """
// Kotlin preprocessing example:
fun preprocessImage(bitmap: Bitmap): FloatArray {
    val resized = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
    val floatArray = FloatArray(3 * 224 * 224)
    
    var idx = 0
    for (c in 0 until 3) {  // RGB channels
        for (y in 0 until 224) {
            for (x in 0 until 224) {
                val pixel = resized.getPixel(x, y)
                val value = when (c) {
                    0 -> Color.red(pixel) / 255.0f
                    1 -> Color.green(pixel) / 255.0f
                    else -> Color.blue(pixel) / 255.0f
                }
                floatArray[idx++] = value
            }
        }
    }
    return floatArray
}
""",
    }

    output_path = Path(output_dir) / "model_config.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Created Android config at {output_path}")
    return str(output_path)


def export_full_pipeline(
    model: nn.Module,
    output_dir: str,
    formats: Tuple[str, ...] = ("onnx", "torchscript"),
    image_size: int = 224,
) -> Dict[str, str]:
    """
    Export model to all requested formats

    Args:
        model: Trained VAE model
        output_dir: Directory for all exports
        formats: Tuple of formats ('onnx', 'torchscript', 'tflite')
        image_size: Input image size

    Returns:
        Dict mapping format to file path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exports = {}

    if "onnx" in formats:
        path = export_to_onnx(model, output_dir / "palmistry_encoder.onnx", image_size)
        exports["onnx"] = path

    if "torchscript" in formats:
        path = export_to_torchscript(
            model, output_dir / "palmistry_encoder.pt", image_size
        )
        exports["torchscript"] = path

    if "tflite" in formats:
        path = export_to_tflite(
            model, output_dir / "palmistry_encoder.tflite", image_size
        )
        if path:
            exports["tflite"] = path

    # Create Android config
    config_path = create_android_config(
        output_dir,
        "palmistry_encoder.onnx",  # Default to ONNX
        model.latent_dim,
        image_size,
    )
    exports["config"] = config_path

    print(f"\nExported models to {output_dir}:")
    for fmt, path in exports.items():
        print(f"  {fmt}: {path}")

    return exports


def verify_onnx_export(onnx_path: str, image_size: int = 224) -> bool:
    """
    Verify ONNX export by running inference

    Args:
        onnx_path: Path to ONNX model
        image_size: Input image size

    Returns:
        True if verification passed
    """
    try:
        import onnxruntime as ort
        import numpy as np

        # Load model
        session = ort.InferenceSession(onnx_path)

        # Get input/output info
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        print(f"Input: {input_name}, shape: {session.get_inputs()[0].shape}")
        print(f"Output: {output_name}, shape: {session.get_outputs()[0].shape}")

        # Run inference
        dummy_input = np.random.randn(1, 3, image_size, image_size).astype(np.float32)
        outputs = session.run([output_name], {input_name: dummy_input})

        print(f"Output shape: {outputs[0].shape}")
        print(" ONNX verification passed!")

        return True

    except Exception as e:
        print(f" ONNX verification failed: {e}")
        return False


if __name__ == "__main__":
    print("Export module loaded successfully")
    print("\nUsage:")
    print("  from export import export_full_pipeline")
    print("  exports = export_full_pipeline(model, './android_model')")
