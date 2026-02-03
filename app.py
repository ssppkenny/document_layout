import os
import sys
import logging
from flask import Flask, render_template, request, send_file, jsonify
from werkzeug.utils import secure_filename
import tempfile
import cv2

# Add the src/ocr_reflow directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'ocr_reflow'))

from main import process_document, process_document_with_layout

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'tif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    try:
        # Validate file upload
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed types: PNG, JPG, JPEG, TIFF, BMP'}), 400

        # Get parameters
        page_width_option = request.form.get('page_width', '2000')
        custom_width = request.form.get('custom_width', '')
        zoom_factor = float(request.form.get('zoom_factor', '2.5'))
        use_layout = request.form.get('use_layout') == 'true'

        # Determine page width
        if page_width_option == 'custom' and custom_width:
            try:
                new_page_width = int(custom_width)
                if new_page_width < 100 or new_page_width > 10000:
                    return jsonify({'error': 'Custom width must be between 100 and 10000'}), 400
            except ValueError:
                return jsonify({'error': 'Invalid custom width value'}), 400
        else:
            try:
                new_page_width = int(page_width_option)
            except ValueError:
                return jsonify({'error': 'Invalid page width value'}), 400

        # Validate zoom factor
        if zoom_factor < 0.1 or zoom_factor > 10:
            return jsonify({'error': 'Zoom factor must be between 0.1 and 10'}), 400

        # Save uploaded file
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], f'input_{filename}')
        file.save(input_path)

        logger.info(f"Processing file: {filename}")
        logger.info(f"Parameters - Page Width: {new_page_width}, Zoom Factor: {zoom_factor}, Use Layout: {use_layout}")

        # Process the document
        try:
            if use_layout:
                result_image = process_document_with_layout(
                    input_path,
                    zoom_factor=zoom_factor,
                    new_page_width=new_page_width
                )
            else:
                result_image = process_document(
                    input_path,
                    zoom_factor=zoom_factor,
                    new_page_width=new_page_width
                )

            # Save result
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'output_{filename}')
            cv2.imwrite(output_path, result_image)

            logger.info(f"Processing complete. Output saved to: {output_path}")

            # Return the processed file
            return send_file(
                output_path,
                mimetype='image/png',
                as_attachment=True,
                download_name=f'reflowed_{filename}'
            )

        except Exception as e:
            logger.error(f"Error processing document: {e}", exc_info=True)
            return jsonify({'error': f'Error processing document: {str(e)}'}), 500
        finally:
            # Clean up input file
            if os.path.exists(input_path):
                try:
                    os.remove(input_path)
                except:
                    pass

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
