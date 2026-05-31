"""
Device utility functions for GPU/MPS acceleration support.

This module provides utilities to detect and use the best available device:
- CUDA (NVIDIA GPU) on Linux/Windows
- MPS (Metal Performance Shaders) on macOS with Apple Silicon
- CPU fallback for other cases
"""

import logging
import platform

logger = logging.getLogger(__name__)


def get_optimal_device():
    """
    Detect and return the best available device for inference.

    Priority:
    1. CUDA (NVIDIA GPU) - Linux/Windows with NVIDIA GPUs
    2. MPS (Metal Performance Shaders) - macOS with Apple Silicon
    3. CPU - fallback for all other cases

    Returns:
        str: Device string ("cuda:0", "mps", or "cpu")
    """
    try:
        import torch
    except ImportError:
        logger.warning("PyTorch not available. Using CPU device.")
        return "cpu"

    # Check for CUDA availability
    if torch.cuda.is_available():
        device = "cuda:0"
        logger.debug(f"CUDA device detected. Using {device}")
        return device

    # Check for MPS availability (macOS with Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        logger.debug("MPS device detected (macOS Apple Silicon). Using mps")
        return device

    # Fallback to CPU
    logger.debug("No GPU acceleration available. Using CPU")
    return "cpu"


def get_device_for_yolo(model):
    """
    Get the optimal device for YOLOv10 model inference.

    This function checks the model's current device and selects the best
    available option for the system.

    Args:
        model: YOLOv10 model instance

    Returns:
        str: Device string for YOLO inference
    """
    optimal_device = get_optimal_device()

    # Check if model is already on a device
    if hasattr(model, 'device'):
        current_device = str(model.device)
        logger.debug(f"Model current device: {current_device}, optimal device: {optimal_device}")

    return optimal_device


def get_device_for_doctr():
    """
    Get the optimal device for DocTR model inference.

    DocTR models use PyTorch backends, so we can use the same detection
    as for general PyTorch models.

    Returns:
        str: Device string for DocTR inference ("cuda:0", "mps", or "cpu")
    """
    optimal_device = get_optimal_device()
    logger.debug(f"Using device for DocTR: {optimal_device}")
    return optimal_device


def print_device_info():
    """
    Print detailed information about available devices on the system.
    Useful for debugging device detection issues.
    """
    try:
        import torch
    except ImportError:
        logger.info("PyTorch not installed. Device detection unavailable.")
        return

    logger.info("=" * 60)
    logger.info("Device Information:")
    logger.info(f"Platform: {platform.system()} {platform.release()}")
    logger.info(f"Python: {platform.python_version()}")
    logger.info(f"PyTorch version: {torch.__version__}")

    # CUDA info
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"  Device {i}: {torch.cuda.get_device_name(i)}")
        logger.info(f"CUDA version: {torch.version.cuda}")

    # MPS info (macOS)
    if hasattr(torch.backends, 'mps'):
        logger.info(f"MPS available: {torch.backends.mps.is_available()}")
        if torch.backends.mps.is_available():
            logger.info("MPS (Metal Performance Shaders) is available on this macOS system")

    # Optimal device
    optimal = get_optimal_device()
    logger.info(f"Optimal device detected: {optimal}")
    logger.info("=" * 60)
