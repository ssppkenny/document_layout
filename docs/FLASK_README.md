# OCR Reflow Web Application

A Flask-based web interface for the OCR Reflow document processing tool.

## Features

- **Web-based interface** - Upload and process documents through your browser
- **Customizable parameters**:
  - Page width: Choose from preset widths (500-4000px) or enter a custom value
  - Zoom factor: Adjust text scaling (0.1 - 10.0)
  - Layout analysis: Optional AI-powered detection of figures, tables, and formulas
- **Drag-and-drop** file upload
- **Automatic download** of processed images
- **Real-time feedback** during processing

## Quick Start

### 1. Install Dependencies

```bash
pixi install
```

### 2. Start the Web Server

```bash
pixi run python app.py
```

Or use the provided script:

```bash
pixi run flask-start
```

### 3. Open in Browser

Navigate to: **http://localhost:5000**

## Usage

1. **Upload Image**: Click or drag-and-drop a document image (PNG, JPG, JPEG, TIFF, BMP)
2. **Set Parameters**:
   - **Page Width**: Select a preset width or choose "Custom" to enter your own
   - **Zoom Factor**: Enter a value between 0.1 and 10 (default: 2.5)
   - **Layout Analysis**: Enable to detect and preserve figures, tables, and formulas
3. **Process**: Click "Process Document" and wait 30-60 seconds
4. **Download**: The processed image will automatically download

## Supported File Types

- PNG (.png)
- JPEG (.jpg, .jpeg)
- TIFF (.tiff, .tif)
- BMP (.bmp)

Maximum file size: 50MB

## Parameters

### Page Width
The width of the output reflowed page in pixels. Predefined options:
- 500, 1000, 1500, 2000 (default), 2500, 3000, 3500, 4000 pixels
- Custom: Enter any value between 100-10,000 pixels

### Zoom Factor
Scaling factor for letters and text:
- Range: 0.1 - 10.0
- Default: 2.5
- Higher values = larger text
- Lower values = smaller text

### Layout Analysis (Experimental)
When enabled, the system will:
- Detect different content types (text, figures, tables, formulas)
- Reflow text blocks while preserving figures and tables as-is
- Requires: `doclayout-yolo` package (already in dependencies)

## API Endpoint

### POST /process

Process a document image with OCR reflow.

**Form Data:**
- `file`: Image file (required)
- `page_width`: Width in pixels or "custom" (required)
- `custom_width`: Custom width value if page_width="custom"
- `zoom_factor`: Float between 0.1-10 (required)
- `use_layout`: "true" to enable layout analysis (optional)

**Response:**
- Success: Returns processed image file
- Error: JSON with error message

**Example with curl:**

```bash
curl -X POST http://localhost:5000/process \
  -F "file=@image.png" \
  -F "page_width=2000" \
  -F "zoom_factor=2.5" \
  -F "use_layout=true" \
  --output reflowed_image.png
```

## Development

### Running in Debug Mode

The Flask app runs in debug mode by default when started with `python app.py`. This enables:
- Auto-reload on code changes
- Detailed error messages
- Debug toolbar

### Production Deployment

For production, use a WSGI server like Gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Troubleshooting

### "Layout analysis is not available"
- Make sure `doclayout-yolo` is installed: `pixi install`
- Check the console logs for import errors

### "Processing takes too long"
- Large images (>5MB) may take 1-2 minutes
- First request after startup takes longer (model loading)
- Subsequent requests are faster due to model caching

### "Out of memory"
- Reduce zoom factor
- Reduce page width
- Use smaller input images

## File Structure

```
document_layout/
├── app.py              # Flask application
├── templates/
│   └── index.html      # Web interface
└── src/ocr_reflow/
    └── main.py         # Core processing functions
```

## License

See LICENSE file in the project root.
