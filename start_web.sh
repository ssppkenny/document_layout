#!/bin/bash

# OCR Reflow Flask Web Server Startup Script

echo "🚀 Starting OCR Reflow Web Application..."
echo ""
echo "Server will be available at: http://localhost:5000"
echo "Press Ctrl+C to stop the server"
echo ""

cd "$(dirname "$0")"
pixi run python app.py
