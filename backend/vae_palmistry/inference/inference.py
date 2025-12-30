import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, List, Tuple, Optional, Dict
from torchvision import transforms


class PalmistryInference:
    """
    Inference wrapper for Palmistry VAE

    Handles:
    - Model loading (PyTorch, ONNX)
    - Image preprocessing
    - Feature extraction
    - Similarity comparison

    Usage:
        inference = PalmistryInference.from_checkpoint('best_model.pt')
        features = inference.extract_features('palm.jpg')
    """

    def __init__(self, model: nn.Module, device: str = "auto", image_size: int = 224):
        """
        Args:
            model: Loaded VAE model
            device: Device for inference
            image_size: Expected input size
        """
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.model.eval()
        self.image_size = image_size

        # Standard transform (must match training!)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        model_class: Optional[type] = None,
        device: str = "auto",
    ) -> "PalmistryInference":
        """
        Load model from checkpoint

        Args:
            checkpoint_path: Path to .pt checkpoint
            model_class: Model class (auto-detected if None)
            device: Device for inference

        Returns:
            PalmistryInference instance
        """
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Get model config
        if "config" in checkpoint:
            latent_dim = checkpoint["config"].get("latent_dim", 256)
        else:
            latent_dim = 256

        # Create model
        if model_class is None:
            # Import here to avoid circular dependency
            import sys

            sys.path.insert(0, str(Path(__file__).parent.parent))
            from models.vae_model import PalmistryVAE

            model_class = PalmistryVAE

        model = model_class(latent_dim=latent_dim)

        # Load weights
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        print(f"Loaded model from {checkpoint_path}")

        return cls(model, device)

    @classmethod
    def from_onnx(cls, onnx_path: str, device: str = "cpu") -> "PalmistryInferenceONNX":
        """
        Load ONNX model for inference

        Args:
            onnx_path: Path to .onnx file
            device: 'cpu' or 'cuda'

        Returns:
            PalmistryInferenceONNX instance
        """
        return PalmistryInferenceONNX(onnx_path, device)

    def preprocess(
        self, image: Union[str, Path, Image.Image, np.ndarray]
    ) -> torch.Tensor:
        """
        Preprocess image for inference

        Args:
            image: Path, PIL Image, or numpy array

        Returns:
            Preprocessed tensor [1, 3, H, W]
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")

        # Transform
        tensor = self.transform(image)

        # Add batch dimension
        return tensor.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def extract_features(
        self, image: Union[str, Path, Image.Image, np.ndarray]
    ) -> np.ndarray:
        """
        Extract feature vector from palm image

        Args:
            image: Input image (path, PIL, or numpy)

        Returns:
            Feature vector [latent_dim]
        """
        tensor = self.preprocess(image)
        features = self.model.encode(tensor)
        return features.cpu().numpy().squeeze()

    @torch.no_grad()
    def extract_features_batch(self, images: List[Union[str, Path]]) -> np.ndarray:
        """
        Extract features from multiple images

        Args:
            images: List of image paths

        Returns:
            Feature matrix [N, latent_dim]
        """
        tensors = []
        for img_path in images:
            tensor = self.preprocess(img_path)
            tensors.append(tensor)

        batch = torch.cat(tensors, dim=0)
        features = self.model.encode(batch)

        return features.cpu().numpy()

    @torch.no_grad()
    def reconstruct(
        self, image: Union[str, Path, Image.Image, np.ndarray]
    ) -> np.ndarray:
        """
        Reconstruct palm image through VAE

        Useful for checking model quality.

        Args:
            image: Input image

        Returns:
            Reconstructed image as numpy array [H, W, 3]
        """
        tensor = self.preprocess(image)
        outputs = self.model(tensor)
        recon = outputs["reconstruction"]

        # Convert to numpy image
        recon = recon.squeeze().cpu().permute(1, 2, 0).numpy()
        recon = (recon * 255).astype(np.uint8)

        return recon

    def compute_similarity(
        self,
        image1: Union[str, np.ndarray],
        image2: Union[str, np.ndarray],
        metric: str = "cosine",
    ) -> float:
        """
        Compute similarity between two palm images

        Args:
            image1: First palm image
            image2: Second palm image
            metric: 'cosine', 'euclidean', or 'manhattan'

        Returns:
            Similarity score (higher = more similar for cosine)
        """
        feat1 = self.extract_features(image1)
        feat2 = self.extract_features(image2)

        if metric == "cosine":
            # Cosine similarity
            similarity = np.dot(feat1, feat2) / (
                np.linalg.norm(feat1) * np.linalg.norm(feat2) + 1e-8
            )
        elif metric == "euclidean":
            # Negative euclidean distance (higher = more similar)
            similarity = -np.linalg.norm(feat1 - feat2)
        elif metric == "manhattan":
            # Negative manhattan distance
            similarity = -np.sum(np.abs(feat1 - feat2))
        else:
            raise ValueError(f"Unknown metric: {metric}")

        return float(similarity)

    def find_most_similar(
        self,
        query_image: Union[str, np.ndarray],
        database_images: List[str],
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Find most similar palms in a database

        Args:
            query_image: Query palm image
            database_images: List of database image paths
            top_k: Number of results to return

        Returns:
            List of (image_path, similarity_score) tuples
        """
        query_feat = self.extract_features(query_image)

        similarities = []
        for img_path in database_images:
            db_feat = self.extract_features(img_path)
            sim = np.dot(query_feat, db_feat) / (
                np.linalg.norm(query_feat) * np.linalg.norm(db_feat) + 1e-8
            )
            similarities.append((img_path, float(sim)))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    @torch.no_grad()
    def analyze_latent(self, image: Union[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Get detailed latent space analysis

        Args:
            image: Input palm image

        Returns:
            Dict with mu, logvar, and importance scores
        """
        tensor = self.preprocess(image)
        mu, logvar = self.model.encode_with_variance(tensor)

        # Compute importance (lower variance = more certain = more important)
        variance = torch.exp(logvar)
        importance = 1.0 / (variance + 1e-8)
        importance = importance / importance.sum()

        return {
            "features": mu.cpu().numpy().squeeze(),
            "mu": mu.cpu().numpy().squeeze(),
            "logvar": logvar.cpu().numpy().squeeze(),
            "variance": variance.cpu().numpy().squeeze(),
            "importance": importance.cpu().numpy().squeeze(),
        }


class PalmistryInferenceONNX:
    """
    ONNX Runtime inference for mobile/production

    Optimized for:
    - CPU inference
    - Low latency
    - Consistent with Android deployment
    """

    def __init__(self, onnx_path: str, device: str = "cpu"):
        """
        Args:
            onnx_path: Path to ONNX model
            device: 'cpu' or 'cuda'
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("Install onnxruntime: pip install onnxruntime")

        # Set providers based on device
        if device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Get input shape
        input_shape = self.session.get_inputs()[0].shape
        self.image_size = input_shape[2]  # Assuming [B, C, H, W]

        # Transform
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ]
        )

        print(f"Loaded ONNX model from {onnx_path}")
        print(f"Input: {self.input_name}, Output: {self.output_name}")

    def preprocess(self, image: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """Preprocess image for ONNX inference"""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")

        tensor = self.transform(image)
        return tensor.unsqueeze(0).numpy()

    def extract_features(
        self, image: Union[str, Image.Image, np.ndarray]
    ) -> np.ndarray:
        """Extract features using ONNX runtime"""
        input_array = self.preprocess(image)
        outputs = self.session.run([self.output_name], {self.input_name: input_array})
        return outputs[0].squeeze()

    def compute_similarity(self, image1, image2, metric: str = "cosine") -> float:
        """Compute similarity between two palms"""
        feat1 = self.extract_features(image1)
        feat2 = self.extract_features(image2)

        if metric == "cosine":
            similarity = np.dot(feat1, feat2) / (
                np.linalg.norm(feat1) * np.linalg.norm(feat2) + 1e-8
            )
        else:
            similarity = -np.linalg.norm(feat1 - feat2)

        return float(similarity)


def demo_inference(model_path: str, image_path: str):
    """
    Demo inference function

    Args:
        model_path: Path to model checkpoint
        image_path: Path to test image
    """
    print("=" * 60)
    print("PALMISTRY VAE INFERENCE DEMO")
    print("=" * 60)

    # Load model
    inference = PalmistryInference.from_checkpoint(model_path)

    # Extract features
    print(f"\nExtracting features from: {image_path}")
    features = inference.extract_features(image_path)

    print(f"Feature vector shape: {features.shape}")
    print(f"Feature range: [{features.min():.3f}, {features.max():.3f}]")
    print(f"Feature mean: {features.mean():.3f}")
    print(f"Feature std: {features.std():.3f}")

    # Analyze latent space
    analysis = inference.analyze_latent(image_path)
    top_dims = np.argsort(analysis["importance"])[-5:][::-1]

    print(f"\nTop 5 most important latent dimensions:")
    for dim in top_dims:
        print(
            f"  Dim {dim}: importance={analysis['importance'][dim]:.4f}, "
            f"value={analysis['features'][dim]:.3f}"
        )

    print("\n Inference completed successfully!")


if __name__ == "__main__":
    print("Inference module loaded successfully")
    print("\nUsage:")
    print("  inference = PalmistryInference.from_checkpoint('model.pt')")
    print("  features = inference.extract_features('palm.jpg')")
