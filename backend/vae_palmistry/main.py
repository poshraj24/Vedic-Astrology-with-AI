import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def train(args):
    """Train the VAE model"""
    from dataloader.dataset import create_dataloaders

    print("=" * 60)
    print("PALMISTRY MODEL TRAINING")
    print("=" * 60)

    # Select model architecture
    if args.hybrid:
        from models.hybrid_model import (
            PalmistryHybridModel,
            PalmistryHybridModelLite,
            count_parameters,
        )
        from training.trainer import HybridModelTrainer as Trainer

        if args.lite:
            print("\nUsing HYBRID LITE model (VAE + U-Net)")
            model = PalmistryHybridModelLite(latent_dim=args.latent_dim)
        else:
            print("\nUsing HYBRID model (VAE + U-Net)")
            model = PalmistryHybridModel(latent_dim=args.latent_dim)
    else:
        from models.vae_model import PalmistryVAE, PalmistryVAELite, count_parameters
        from training.trainer import VAETrainer as Trainer

        if args.lite:
            print("\nUsing VAE LITE model")
            model = PalmistryVAELite(latent_dim=args.latent_dim)
        else:
            print("\nUsing standard VAE model")
            model = PalmistryVAE(latent_dim=args.latent_dim)

    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Latent dimension: {args.latent_dim}")

    # Create data loaders
    print(f"\nLoading data from: {args.data_dir}")
    train_loader, val_loader = create_dataloaders(
        train_dir=args.data_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
    )

    # Training config
    config = {
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "beta_start": args.beta_start,
        "beta_end": args.beta_end,
        "beta_warmup_epochs": args.beta_warmup,
        "patience": args.patience,
        "checkpoint_dir": args.checkpoint_dir,
        "log_dir": args.log_dir,
        "save_every": args.save_every,
    }

    # Create trainer
    trainer = Trainer(model, config, device=args.device)

    # Train
    history = trainer.train(train_loader, val_loader)

    # Visualize reconstructions
    if val_loader:
        vis_path = Path(args.checkpoint_dir) / "reconstructions.png"
        trainer.visualize_reconstructions(val_loader, save_path=str(vis_path))

    print("\n Training completed!")
    print(f"  Checkpoints saved to: {args.checkpoint_dir}")
    print(f"  Best model: {args.checkpoint_dir}/best_model.pt")

    return history


def export(args):
    """Export model for mobile deployment"""
    from export.export_mobile import export_full_pipeline, verify_onnx_export
    import torch

    print("=" * 60)
    print("EXPORTING MODEL FOR ANDROID")
    print("=" * 60)

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    # Determine model type and latent dim
    if "config" in checkpoint:
        latent_dim = checkpoint["config"].get("latent_dim", args.latent_dim)
    else:
        latent_dim = args.latent_dim

    # Create model based on type
    if args.hybrid:
        from models.hybrid_model import PalmistryHybridModel, PalmistryHybridModelLite

        if args.lite:
            model = PalmistryHybridModelLite(latent_dim=min(latent_dim, 128))
        else:
            model = PalmistryHybridModel(latent_dim=latent_dim)
    else:
        from models.vae_model import PalmistryVAE, PalmistryVAELite

        if args.lite:
            model = PalmistryVAELite(latent_dim=min(latent_dim, 128))
        else:
            model = PalmistryVAE(latent_dim=latent_dim)

    # Load weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # Export
    formats = tuple(args.formats.split(","))
    print(f"\nExporting to formats: {formats}")

    exports = export_full_pipeline(
        model, args.output_dir, formats=formats, image_size=args.image_size
    )

    # Verify ONNX if exported
    if "onnx" in exports:
        print("\nVerifying ONNX export...")
        verify_onnx_export(exports["onnx"], args.image_size)

    print("\n Export completed!")
    print(f"  Models saved to: {args.output_dir}")
    print("\n  Copy these files to your Android project:")
    for fmt, path in exports.items():
        print(f"    - {path}")


def infer(args):
    """Run inference on palm images"""
    from inference.inference import PalmistryInference
    import numpy as np

    print("=" * 60)
    print("PALMISTRY VAE INFERENCE")
    print("=" * 60)

    # Load model
    print(f"\nLoading model: {args.checkpoint}")
    inference = PalmistryInference.from_checkpoint(args.checkpoint, device=args.device)

    if args.image:
        # Single image inference
        print(f"\nProcessing: {args.image}")

        features = inference.extract_features(args.image)
        print(f"\nFeature vector shape: {features.shape}")
        print(f"Feature statistics:")
        print(f"  Min: {features.min():.4f}")
        print(f"  Max: {features.max():.4f}")
        print(f"  Mean: {features.mean():.4f}")
        print(f"  Std: {features.std():.4f}")

        # Save features if requested
        if args.output:
            np.save(args.output, features)
            print(f"\nFeatures saved to: {args.output}")

        # Latent analysis
        if args.analyze:
            analysis = inference.analyze_latent(args.image)
            top_dims = np.argsort(analysis["importance"])[-10:][::-1]

            print(f"\nTop 10 most important latent dimensions:")
            for dim in top_dims:
                print(
                    f"  Dim {dim:3d}: importance={analysis['importance'][dim]:.4f}, "
                    f"value={analysis['features'][dim]:+.3f}, "
                    f"variance={analysis['variance'][dim]:.4f}"
                )

    elif args.compare:
        # Compare two images
        img1, img2 = args.compare
        print(f"\nComparing:")
        print(f"  Image 1: {img1}")
        print(f"  Image 2: {img2}")

        similarity = inference.compute_similarity(img1, img2, metric="cosine")
        print(f"\nCosine similarity: {similarity:.4f}")

        if similarity > 0.9:
            print("  → Very similar palms (possibly same person)")
        elif similarity > 0.7:
            print("  → Similar palms")
        elif similarity > 0.5:
            print("  → Somewhat similar")
        else:
            print("  → Different palms")

    elif args.batch_dir:
        # Batch inference
        print(f"\nProcessing directory: {args.batch_dir}")

        image_paths = list(Path(args.batch_dir).glob("*.jpg")) + list(
            Path(args.batch_dir).glob("*.png")
        )

        print(f"Found {len(image_paths)} images")

        features = inference.extract_features_batch([str(p) for p in image_paths])
        print(f"Extracted features shape: {features.shape}")

        if args.output:
            np.savez(
                args.output, features=features, paths=[str(p) for p in image_paths]
            )
            print(f"\nFeatures saved to: {args.output}")

    print("\n Inference completed!")


def main():
    parser = argparse.ArgumentParser(
        description="Palmistry VAE - Self-supervised palm feature learning"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # =========================================================================
    # Train command
    # =========================================================================
    train_parser = subparsers.add_parser("train", help="Train the VAE model")
    train_parser.add_argument(
        "--data-dir", type=str, required=True, help="Directory containing palm images"
    )
    train_parser.add_argument(
        "--val-dir",
        type=str,
        default=None,
        help="Separate validation directory (optional)",
    )
    train_parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    train_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    train_parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    train_parser.add_argument(
        "--latent-dim", type=int, default=256, help="Latent space dimension"
    )
    train_parser.add_argument(
        "--image-size", type=int, default=224, help="Input image size"
    )
    train_parser.add_argument(
        "--beta-start", type=float, default=0.0, help="Initial beta for KL loss"
    )
    train_parser.add_argument(
        "--beta-end", type=float, default=1.0, help="Final beta for KL loss"
    )
    train_parser.add_argument(
        "--beta-warmup", type=int, default=10, help="Beta warmup epochs"
    )
    train_parser.add_argument(
        "--patience", type=int, default=15, help="Early stopping patience"
    )
    train_parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Directory to save checkpoints",
    )
    train_parser.add_argument(
        "--log-dir", type=str, default="./logs", help="Directory for tensorboard logs"
    )
    train_parser.add_argument(
        "--save-every", type=int, default=10, help="Save checkpoint every N epochs"
    )
    train_parser.add_argument(
        "--num-workers", type=int, default=4, help="DataLoader workers"
    )
    train_parser.add_argument(
        "--val-split", type=float, default=0.1, help="Validation split ratio"
    )
    train_parser.add_argument(
        "--device", type=str, default="auto", help="Device (auto/cuda/cpu/mps)"
    )
    train_parser.add_argument(
        "--lite", action="store_true", help="Use lightweight model for mobile"
    )
    train_parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Use hybrid VAE + U-Net model for line segmentation",
    )

    # =========================================================================
    # Export command
    # =========================================================================
    export_parser = subparsers.add_parser("export", help="Export model for Android")
    export_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    export_parser.add_argument(
        "--output-dir",
        type=str,
        default="./android_model",
        help="Output directory for exported models",
    )
    export_parser.add_argument(
        "--formats",
        type=str,
        default="onnx,torchscript",
        help="Export formats (comma-separated: onnx,torchscript,tflite)",
    )
    export_parser.add_argument(
        "--image-size", type=int, default=224, help="Input image size"
    )
    export_parser.add_argument(
        "--latent-dim",
        type=int,
        default=256,
        help="Latent dimension (if not in checkpoint)",
    )
    export_parser.add_argument(
        "--lite", action="store_true", help="Export lightweight model"
    )
    export_parser.add_argument(
        "--hybrid", action="store_true", help="Export hybrid VAE + U-Net model"
    )

    # =========================================================================
    # Inference command
    # =========================================================================
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    infer_parser.add_argument(
        "--image", type=str, default=None, help="Single image to process"
    )
    infer_parser.add_argument(
        "--compare",
        type=str,
        nargs=2,
        default=None,
        help="Two images to compare similarity",
    )
    infer_parser.add_argument(
        "--batch-dir",
        type=str,
        default=None,
        help="Directory of images for batch inference",
    )
    infer_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for features (.npy or .npz)",
    )
    infer_parser.add_argument(
        "--analyze", action="store_true", help="Show detailed latent analysis"
    )
    infer_parser.add_argument(
        "--device", type=str, default="auto", help="Device (auto/cuda/cpu/mps)"
    )

    # Parse arguments
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        print("\n" + "=" * 60)
        print("QUICK START")
        print("=" * 60)
        print(
            """
1. TRAIN the model on your palm dataset:
   
   # Standard VAE (holistic features only)
   python main.py train --data-dir /path/to/palm/images --epochs 100
   
   # Hybrid VAE + U-Net (features + line segmentation)
   python main.py train --data-dir /path/to/palm/images --epochs 100 --hybrid

2. EXPORT for Android deployment:
   python main.py export --checkpoint ./checkpoints/best_model.pt
   python main.py export --checkpoint ./checkpoints/best_model.pt --hybrid

3. RUN INFERENCE on new images:
   python main.py infer --checkpoint ./checkpoints/best_model.pt --image palm.jpg

For more options, use: python main.py <command> --help
        """
        )
        return

    # Execute command
    if args.command == "train":
        train(args)
    elif args.command == "export":
        export(args)
    elif args.command == "infer":
        infer(args)


if __name__ == "__main__":
    main()
