# Quick Test for Jupyter Notebook

This is a minimal test to verify that the ocr-reflow package works in Jupyter.

## Step 1: Test Import

Run this in a Jupyter cell:

```python
# Test basic import
from ocr_reflow import process_document, Letter
print("✓ Package imported successfully!")

# Check version
import ocr_reflow
print(f"Version: {ocr_reflow.__version__}")
```

Expected output:
```
✓ Package imported successfully!
Version: 0.1.0
```

## Step 2: Test Dependencies

```python
# Test all dependencies
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import spatial
import shapely
from doctr.models import detection_predictor

print("✓ All dependencies loaded successfully!")
```

## Step 3: Test Letter Class

```python
# Test the Letter dataclass
from ocr_reflow import Letter

letter = Letter(xmin=10, ymin=20, xmax=30, ymax=40, bl=5)
print(f"Letter: {letter}")
print(f"  Position: ({letter.xmin}, {letter.ymin}) to ({letter.xmax}, {letter.ymax})")
print(f"  Baseline: {letter.bl}")
print("✓ Letter class works!")
```

## Step 4: Test with a Sample Image (Optional)

If you have a document image available:

```python
from ocr_reflow import process_document
import cv2
from matplotlib import pyplot as plt

# Process a document (replace with your image path)
image_path = "path/to/your/document.png"
result = process_document(image_path)

# Display the result
plt.figure(figsize=(10, 12))
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Reflowed Document')
plt.show()

print(f"✓ Processed document successfully!")
print(f"  Output size: {result.shape}")
```

## Common Issues

### Module Not Found

If you get `ModuleNotFoundError: No module named 'ocr_reflow'`:

1. Make sure you installed the package:
   ```bash
   pip install -e .
   ```

2. Restart the Jupyter kernel:
   - Click "Kernel" → "Restart Kernel"

3. If still not working, check your Python path:
   ```python
   import sys
   print(sys.executable)
   print('\n'.join(sys.path))
   ```

### Wrong Kernel

Make sure Jupyter is using the same Python environment where you installed the package:

```python
import sys
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
```

This should point to your virtual environment or pixi environment.

## Success!

If all steps passed, you're ready to use the package! Check out:
- `notebooks/example_usage.ipynb` - Complete example notebook
- `docs/JUPYTER_GUIDE.md` - Detailed Jupyter usage guide
- `README.md` - Full documentation
