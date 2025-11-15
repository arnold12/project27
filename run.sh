#!/bin/bash
# Startup script for Policy Document Summarization Assistant

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Create data directories if they don't exist
mkdir -p data/documents data/chunks data/faiss_index

# Run the application
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

