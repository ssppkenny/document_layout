#!/usr/bin/env python3
"""
Check if image has EXIF orientation that might cause rotation
"""

import cv2
from PIL import Image

def check_image_orientation(image_path='images/jtg_p033.png'):
    """Check image orientation and EXIF data"""

    print("=" * 70)
    print("CHECKING IMAGE ORIENTATION")
    print("=" * 70)

    # Check with PIL
    try:
        pil_img = Image.open(image_path)
        print(f"\nPIL Image Info:")
        print(f"  Size: {pil_img.size}")
        print(f"  Mode: {pil_img.mode}")
        print(f"  Format: {pil_img.format}")

        # Check EXIF
        if hasattr(pil_img, '_getexif') and pil_img._getexif():
            exif = pil_img._getexif()
            orientation = exif.get(274)  # 274 is the EXIF orientation tag
            print(f"  EXIF Orientation: {orientation}")
            if orientation:
                orientations = {
                    1: "Normal",
                    2: "Mirrored",
                    3: "Rotated 180°",
                    4: "Mirrored and rotated 180°",
                    5: "Mirrored and rotated 90° CCW",
                    6: "Rotated 90° CW",
                    7: "Mirrored and rotated 90° CW",
                    8: "Rotated 90° CCW"
                }
                print(f"  Meaning: {orientations.get(orientation, 'Unknown')}")
        else:
            print(f"  No EXIF orientation data")
    except Exception as e:
        print(f"  Error reading with PIL: {e}")

    # Check with OpenCV
    cv_img = cv2.imread(image_path)
    print(f"\nOpenCV Image Info:")
    print(f"  Shape: {cv_img.shape} (H x W x C)")
    print(f"  Data type: {cv_img.dtype}")

    print("\n" + "=" * 70)

if __name__ == '__main__':
    check_image_orientation()
