#!/bin/bash
set -e
cd "$(dirname "$0")"
export MAIT_ENABLE_RUNNER=1
echo "Opening MAIT on this computer at http://localhost:8501"
python -m streamlit run streamlit_app.py --server.port 8501 --server.address 127.0.0.1
