# Quick Start Guide

This guide will get you up and running with the text segmentation and reflow project in under 5 minutes.

## Step 1: Install Pixi

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

After installation, restart your terminal or run:
```bash
source ~/.bashrc  # or ~/.zshrc depending on your shell
```

## Step 2: Clone and Setup

```bash
# Navigate to your projects directory
cd ~/code/python/segmentation  # or wherever your project is

# Install all dependencies (this may take a few minutes the first time)
pixi install
```

## Step 3: Activate Environment

```bash
pixi shell
```

You should see your prompt change to indicate you're in the Pixi environment.

## Step 4: Run Your First Example

```bash
cd src
python main.py ../test_economist_january_original.png
```

This will:
1. Detect all text in the image
2. Extract individual characters
3. Group them into lines
4. Reflow the text onto a new page
5. Save the result as `out.png`

## Step 5: View Results

```bash
# View the reflowed page
xdg-open out.png  # Linux
# or
open out.png      # macOS

# View debug outputs
xdg-open out1.png  # Character bounding boxes
xdg-open out2.png  # Detected lines
```

## What's Next?

### Try Different Images

```bash
python main.py path/to/your/image.png
```

### Adjust Parameters

Edit the `zoom_factor` and `new_page_width` in `main.py`:

```python
# In main.py, near the bottom:
zoom_factor = 2.0  # Make text bigger
new_page_width = 1000  # Wider page
```

### Run Tests

```bash
cd ..
python test_1950.py
python test_outlier_spacing.py
```

### Explore in Jupyter

```bash
jupyter lab
# Open segmentation.ipynb
```

## Common Commands

```bash
# Activate environment
pixi shell

# Install new package
pixi add package-name

# Update dependencies
pixi update

# List installed packages
pixi list

# Leave environment
exit
```

## Troubleshooting

### "pixi: command not found"

Make sure you've restarted your terminal after installing Pixi, or run:
```bash
source ~/.bashrc
```

### "CUDA not available"

The project requires CUDA 13.0. Check with:
```bash
nvidia-smi
```

If you don't have CUDA, you can still run on CPU (will be slower).

### Import errors

Make sure you're in the Pixi environment:
```bash
pixi shell
cd src
python main.py <image>
```

## Next Steps

- Read the full [README.md](../README.md) for detailed documentation
- Check out the test scripts to see examples
- Modify `main.py` to suit your needs
- Explore `reflow.py` to understand the text layout algorithms

---

**Questions?** Check the README.md or open an issue.
