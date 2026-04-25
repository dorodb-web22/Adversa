#!/bin/bash
# Start backend on 8000
uvicorn server.app:app --host 0.0.0.0 --port 8000 &
# Wait for backend
sleep 3
# Start frontend on 7860 (expected by HF Spaces)
python frontend/app.py
