"""
Sauvola's Method for Document Binarization

Implementation of Sauvola's adaptive binarization algorithm for document images.
This method improves upon Niblack's algorithm by incorporating the dynamic range
of the image's gray value standard deviation.

Reference:
Sauvola, J., & Pietikäinen, M. (2000). Adaptive document image binarization.
Pattern Recognition, 33(2), 225-236.

The threshold is calculated as:
    T_sauvola = m * (1 - k * (1 - S/R))

where:
    m = local mean of the window
    S = local standard deviation of the window
    k = parameter (default 0.5)
    R = dynamic range of standard deviation (default 128 for grayscale images)
"""

import cv2
import numpy as np


def normalize_image(image: np.ndarray, alpha: float = 0, beta: float = 255, norm_type: int = cv2.NORM_MINMAX) -> np.ndarray:
    """
    Normalize image pixel intensity values to a specified range.

    Normalization brings the image to a range that is normal to sense by adjusting
    the pixel intensity values. This can improve the performance of subsequent
    image processing operations like binarization.

    Args:
        image: Input image (grayscale or BGR)
        alpha: Lower boundary of the output range (default: 0)
        beta: Upper boundary of the output range (default: 255)
        norm_type: Normalization type. Options:
                  - cv2.NORM_MINMAX: Normalizes to [alpha, beta] range (default)
                  - cv2.NORM_L1: L1 normalization
                  - cv2.NORM_L2: L2 normalization
                  - cv2.NORM_INF: Infinity normalization

    Returns:
        Normalized image (uint8)
    """
    # Convert to float for normalization
    img_float = image.astype(np.float32)

    # Apply normalization
    normalized = cv2.normalize(img_float, None, alpha, beta, norm_type)

    # Convert back to uint8
    return normalized.astype(np.uint8)


def sauvola_binarization(
    image: np.ndarray,
    window_size: int = 15,
    k: float = 0.5,
    R: float = 128.0
) -> np.ndarray:
    """
    Apply Sauvola's adaptive binarization to a grayscale image.

    Args:
        image: Input grayscale image (uint8)
        window_size: Size of the local window (must be odd). Default: 15
        k: Sauvola parameter controlling the effect of standard deviation. Default: 0.5
        R: Dynamic range of standard deviation. Default: 128 for 8-bit images

    Returns:
        Binary image (uint8) where foreground is 255 and background is 0
    """
    # Ensure image is grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Ensure window size is odd
    if window_size % 2 == 0:
        window_size += 1

    # Convert to float for calculations
    img_float = image.astype(np.float64)

    # Calculate local mean using box filter (efficient convolution)
    mean = cv2.boxFilter(img_float, cv2.CV_64F, (window_size, window_size))

    # Calculate local standard deviation
    # Var(X) = E[X^2] - E[X]^2
    mean_squared = cv2.boxFilter(img_float ** 2, cv2.CV_64F, (window_size, window_size))
    variance = mean_squared - mean ** 2

    # Handle numerical issues (variance should be non-negative)
    variance = np.maximum(variance, 0)
    std_dev = np.sqrt(variance)

    # Calculate Sauvola threshold
    # T = m * (1 - k * (1 - S/R))
    # T = m * (1 - k + k*S/R)
    threshold = mean * (1.0 - k * (1.0 - std_dev / R))

    # Apply threshold: pixel is foreground (0=black) if below threshold
    # Background is white (255)
    # This produces black text on white background
    binary = np.ones_like(image) * 255
    binary[img_float < threshold] = 0

    return binary.astype(np.uint8)


def niblack_binarization(
    image: np.ndarray,
    window_size: int = 15,
    k: float = -0.2
) -> np.ndarray:
    """
    Apply Niblack's adaptive binarization to a grayscale image.

    The threshold is calculated as:
        T_niblack = m + k * S

    where:
        m = local mean of the window
        S = local standard deviation of the window
        k = Niblack parameter (default -0.2)

    Args:
        image: Input grayscale image (uint8)
        window_size: Size of the local window (must be odd). Default: 15
        k: Niblack parameter. Default: -0.2

    Returns:
        Binary image (uint8) where foreground is 255 and background is 0
    """
    # Ensure image is grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Ensure window size is odd
    if window_size % 2 == 0:
        window_size += 1

    # Convert to float for calculations
    img_float = image.astype(np.float64)

    # Calculate local mean
    mean = cv2.boxFilter(img_float, cv2.CV_64F, (window_size, window_size))

    # Calculate local standard deviation
    mean_squared = cv2.boxFilter(img_float ** 2, cv2.CV_64F, (window_size, window_size))
    variance = mean_squared - mean ** 2
    variance = np.maximum(variance, 0)
    std_dev = np.sqrt(variance)

    # Calculate Niblack threshold
    threshold = mean + k * std_dev

    # Apply threshold: pixel is foreground (0=black) if below threshold
    # Background is white (255)
    binary = np.ones_like(image) * 255
    binary[img_float < threshold] = 0

    return binary.astype(np.uint8)


def otsu_binarization(image: np.ndarray) -> np.ndarray:
    """
    Apply Otsu's global binarization to a grayscale image.

    Otsu's method automatically determines the optimal threshold value
    by maximizing the between-class variance. This is a global thresholding
    method (no window_size parameter needed).

    Reference:
    Otsu, N. (1979). A threshold selection method from gray-level histograms.
    IEEE Transactions on Systems, Man, and Cybernetics, 9(1), 62-66.

    Args:
        image: Input grayscale image (uint8)

    Returns:
        Binary image (uint8) where foreground is 255 (white) and background is 0 (black)
    """
    # Ensure image is grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding
    # cv2.threshold returns (threshold_value, binary_image)
    # For documents with dark text (low pixel values ~0) on light background (high pixel values ~255):
    # - Otsu finds a threshold value between them
    # - THRESH_BINARY: pixels > threshold → white (255), pixels <= threshold → black (0)
    # - Since text has LOW values (< threshold), it becomes black (0)
    # - Since background has HIGH values (> threshold), it becomes white (255)
    # This gives us black text on white background, which is what we want
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary


def binarize_document(
    image: np.ndarray,
    method: str = "sauvola",
    window_size: int = 15,
    k: float = None,
    R: float = 128.0
) -> np.ndarray:
    """
    Apply adaptive binarization to a document image.

    Args:
        image: Input image (grayscale or BGR)
        method: Binarization method - "sauvola", "niblack", or "otsu". Default: "sauvola"
        window_size: Size of the local window (must be odd). Default: 15
                    (Not used for Otsu method)
        k: Method-specific parameter. If None, uses default for the method:
           - Sauvola: 0.5
           - Niblack: -0.2
           - Otsu: ignored (threshold is automatically determined)
        R: Dynamic range for Sauvola method. Default: 128

    Returns:
        Binary image (uint8) where foreground is 0 (black) and background is 255 (white)
    """
    method = method.lower()

    if method == "sauvola":
        if k is None:
            k = 0.5
        return sauvola_binarization(image, window_size, k, R)
    elif method == "niblack":
        if k is None:
            k = -0.2
        return niblack_binarization(image, window_size, k)
    elif method == "otsu":
        return otsu_binarization(image)
    else:
        raise ValueError(f"Unknown binarization method: {method}. Use 'sauvola', 'niblack', or 'otsu'")


def adaptive_window_size(image: np.ndarray) -> int:
    """
    Calculate an adaptive window size based on image dimensions.

    A good window size should be related to the expected character size.
    This function estimates a reasonable window size as approximately
    1/50th of the image width, constrained to be odd and within reasonable bounds.

    Args:
        image: Input image

    Returns:
        Odd window size value between 5 and 101
    """
    h, w = image.shape[:2]
    # Use ~2% of image width as window size
    window_size = int(w * 0.02)

    # Make it odd
    if window_size % 2 == 0:
        window_size += 1

    # Constrain to reasonable range
    window_size = max(5, min(101, window_size))

    return window_size


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python binarization.py <image_path> [output_path] [method] [window_size]")
        print("  method: sauvola (default), niblack, or otsu")
        print("  window_size: integer (default: auto-calculated, not used for otsu)")
        sys.exit(1)

    # Read image
    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Error: Could not read image {sys.argv[1]}")
        sys.exit(1)

    # Get parameters
    output_path = sys.argv[2] if len(sys.argv) > 2 else "output_binarized.png"
    method = sys.argv[3] if len(sys.argv) > 3 else "sauvola"

    if len(sys.argv) > 4:
        window_size = int(sys.argv[4])
    else:
        window_size = adaptive_window_size(img)
        print(f"Auto-calculated window size: {window_size}")

    # Apply binarization
    print(f"Applying {method} binarization with window size {window_size}...")
    binary = binarize_document(img, method=method, window_size=window_size)

    # Save result
    cv2.imwrite(output_path, binary)
    print(f"Saved binarized image to {output_path}")
