# Quick Start Guide for Jupyter Notebooks

This guide will help you get started with using the `ocr-reflow` package in Jupyter notebooks.

## Installation

### Step 1: Install the Package

From the project root directory, install the package in editable mode:

```bash
# If using pixi (recommended)
pixi shell
/home/sergey/code/python/segmentation/.pixi/envs/default/bin/pip install -e .

# Or if using a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

The `[dev]` option installs additional development dependencies including Jupyter.

### Step 2: Start Jupyter

```bash
# If using pixi
pixi run jupyter lab

# Or if using venv
jupyter lab
```

## Basic Usage in Jupyter

### 1. Import the Package

```python
from ocr_reflow import process_document
import cv2
from matplotlib import pyplot as plt
```

### 2. Process a Document

```python
# Path to your document image
image_path = "path/to/your/document.png"

# Process the document
result = process_document(image_path)
```

### 3. Display the Result

```python
# Convert BGR to RGB for matplotlib
result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

# Display
plt.figure(figsize=(12, 16))
plt.imshow(result_rgb)
plt.axis('off')
plt.title('Reflowed Document')
plt.show()
```

### 4. Save the Result

```python
# Save to file
cv2.imwrite("output_reflowed.png", result)
print("Saved to output_reflowed.png")
```

## Example Notebook

Check out the complete example notebook at `notebooks/example_usage.ipynb` for more detailed examples including:

- Side-by-side comparison of original and reflowed documents
- Batch processing multiple images
- Advanced usage with individual components

## Common Issues

### Import Error

If you get an import error, make sure you:
1. Installed the package with `pip install -e .`
2. Are running Jupyter from the same environment where you installed the package

### Package Not Found

If Jupyter can't find the package:
```python
import sys
sys.path.insert(0, '/path/to/segmentation/src')
from ocr_reflow import process_document
```

### CUDA/GPU Issues

The package will work with or without GPU. If you have CUDA issues, the CPU version will be used automatically.

## Tips

1. **Large Images**: Processing large images may take some time. Consider resizing very large images first.

2. **Memory**: The package loads the entire image into memory. For very large documents, monitor your memory usage.

3. **Visualization**: Use `%matplotlib inline` at the top of your notebook for inline plots.

4. **Reload Changes**: If you modify the package code, restart the Jupyter kernel to see changes (or use autoreload):
   ```python
   %load_ext autoreload
   %autoreload 2
   ```

## Next Steps

- Explore the example notebook: `notebooks/example_usage.ipynb`
- Check out the API documentation in the main [README.md](../README.md)
- Try processing your own documents!
