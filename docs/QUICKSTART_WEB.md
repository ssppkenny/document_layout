# 🚀 Quick Start Guide - OCR Reflow Web Application

## Installation Complete! ✅

The Flask web application has been successfully set up with the following features:

### ✨ Features
- **Web-based interface** with drag-and-drop file upload
- **Page width dropdown** with preset values: 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000 pixels + custom option
- **Zoom factor input** for custom text scaling (0.1 - 10.0)
- **Layout analysis option** for intelligent handling of figures and tables
- **Automatic file download** after processing

---

## 🎯 How to Start the Web Server

### Option 1: Using Pixi (Recommended)
```bash
pixi run web
```

### Option 2: Using the Shell Script
```bash
./start_web.sh
```

### Option 3: Direct Python
```bash
pixi run python app.py
```

---

## 📖 Using the Web Interface

1. **Open your browser** and navigate to:
   ```
   http://localhost:5000
   ```

2. **Upload an image**:
   - Click the upload area or drag & drop
   - Supported formats: PNG, JPG, JPEG, TIFF, BMP
   - Max size: 50MB

3. **Configure parameters**:
   - **Page Width**: Select from dropdown (500-4000px) or choose "Custom"
   - **Zoom Factor**: Enter value (default: 2.5)
   - **Layout Analysis**: Check to enable AI-powered layout detection

4. **Process**:
   - Click "Process Document"
   - Wait 30-60 seconds (first run may take longer)
   - File will automatically download when complete

---

## 🔧 Command Line Alternative

You can still use the command-line interface:

```bash
# Basic usage
pixi run python src/ocr_reflow/main.py images/dvurog_p025.png

# With custom parameters
pixi run python src/ocr_reflow/main.py images/dvurog_p025.png \
  --page-width 1500 \
  --zoom-factor 3.0

# With layout analysis
pixi run python src/ocr_reflow/main.py images/dvurog_p025.png \
  --layout \
  --page-width 2000 \
  --zoom-factor 2.5
```

---

## 📁 Project Structure

```
document_layout/
├── app.py                  # Flask web application
├── start_web.sh           # Startup script
├── templates/
│   └── index.html         # Web interface
├── src/ocr_reflow/
│   ├── main.py            # Core processing
│   ├── reflow.py          # Reflow logic
│   ├── layout.py          # Layout analysis
│   └── device_utils.py    # Device detection
├── FLASK_README.md        # Detailed Flask documentation
└── pixi.toml              # Dependencies
```

---

## 🎨 Web Interface Features

### Page Width Options
- **500px** - Very narrow, mobile-friendly
- **1000px** - Narrow
- **1500px** - Medium
- **2000px** - Default, good for most documents
- **2500px** - Large
- **3000px** - Very large
- **3500px** - Extra large
- **4000px** - Maximum preset
- **Custom** - Enter any value (100-10,000px)

### Zoom Factor
- Controls text size scaling
- Range: 0.1 to 10.0
- Default: 2.5
- Examples:
  - 1.0 = Original size
  - 2.5 = 2.5× larger (default)
  - 5.0 = Very large text
  - 0.5 = Half size

### Layout Analysis
- Uses AI (doclayout-yolo) to detect:
  - Plain text blocks
  - Titles/headings
  - Figures/images
  - Tables
  - Formulas
- Text blocks are reflowed
- Figures/tables are preserved as-is

---

## 🐛 Troubleshooting

### Server won't start
```bash
# Reinstall dependencies
pixi install

# Check if port 5000 is available
lsof -i :5000
```

### Processing fails
- Check file format (must be image: PNG, JPG, etc.)
- Verify file size (< 50MB)
- Check console logs for errors
- Try with layout analysis disabled first

### Slow processing
- First request after startup is slower (model loading)
- Large images take longer
- Layout analysis adds ~10-20 seconds
- Subsequent requests are faster (model cached)

---

## 📚 Documentation

- **Flask Web App**: See `FLASK_README.md`
- **Command Line**: Run `pixi run python src/ocr_reflow/main.py --help`
- **API Documentation**: See `FLASK_README.md` for API endpoint details

---

## 🎉 You're All Set!

Start the server with:
```bash
pixi run web
```

Then open: **http://localhost:5000**

Enjoy your OCR Reflow web application! 🎊
