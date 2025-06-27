#!/bin/bash

# MAIT Streamlit App Launcher
echo "Starting MAIT Pipeline Configuration Interface..."

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "Streamlit not found. Installing required packages..."
    pip install -r streamlit_requirements.txt
fi

# Create necessary directories
mkdir -p configs
mkdir -p generated_notebooks

# Launch the Streamlit app
echo "Launching MAIT Configuration Interface..."
echo "Open your browser and go to: http://localhost:8501"
echo "Use Ctrl+C to stop the application"

streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
