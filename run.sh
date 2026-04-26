#!/bin/bash
set -e

echo "🚀 Starting Adversa — Multi-Agent Courtroom Simulator"
echo "=================================================="

# Start FastAPI backend on port 8000 (internal)
echo "[1/2] Starting FastAPI environment on port 8000..."
uvicorn server.app:app --host 0.0.0.0 --port 8000 --log-level info &
BACKEND_PID=$!

# Wait until backend is healthy
echo "      Waiting for backend to be ready..."
for i in {1..20}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "      ✅ Backend ready!"
        break
    fi
    sleep 1
done

# Start Gradio frontend on port 7860 (HF Spaces public port)
echo "[2/2] Starting Gradio dashboard on port 7860..."
python frontend/app.py

# If Gradio exits, kill backend
kill $BACKEND_PID 2>/dev/null || true
