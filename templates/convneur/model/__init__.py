"""
ConvNeur Model Builder Factory
Provides standard model configurations (m1, m2, m3, m4) scaled for both:
- ImageNet (img_size >= 128): chunk_size = [196, 196, 196, 49]
- CIFAR / Small Scale (img_size < 128): chunk_size = [16, 16, 4, 1]
"""

from model.convneur import ConvNeur


def build_model(scale: str = "m1", num_classes: int = 10, img_size: int = 32) -> ConvNeur:
    """
    Factory function for ConvNeur models.
    Supports ImageNet (img_size >= 128) and CIFAR-10 / small scale (img_size < 128).
    """
    if img_size >= 128:
        chunk_size = [196, 196, 196, 49]
        use_mem_batch = True
    else:
        chunk_size = [16, 16, 4, 1]
        use_mem_batch = False

    scales = {
        "m1": {"depths": [2, 2, 6, 2], "dims": [40, 80, 160, 320], "mem_dim": 80},
        "m2": {"depths": [2, 2, 6, 2], "dims": [48, 96, 192, 384], "mem_dim": 96},
        "m3": {"depths": [2, 2, 6, 2], "dims": [64, 128, 256, 512], "mem_dim": 128},
        "m4": {"depths": [2, 2, 8, 2], "dims": [80, 160, 320, 640], "mem_dim": 160},
    }

    if scale not in scales:
        raise ValueError(f"Scale {scale} not supported. Choices: {list(scales.keys())}")

    cfg = scales[scale]
    return ConvNeur(
        num_classes=num_classes,
        depths=cfg["depths"],
        dims=cfg["dims"],
        mem_dim=cfg["mem_dim"],
        chunk_size=chunk_size,
        use_mem_batch_size=use_mem_batch,
    )
