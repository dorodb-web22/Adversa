# Adversa Gradio Dashboard Setup

1. **Local Setup:**
   Run the Gradio dashboard locally with:
   ```bash
   python frontend/app.py
   ```
   This will start the UI on `http://localhost:7865`. It connects to the FastAPI backend on `http://localhost:7860`.

2. **HuggingFace Spaces Setup:**
   When you create the HuggingFace Space (Docker), it expects port 7860. The current Dockerfile runs the FastAPI backend. 
   
   To run **both** the backend and the frontend on HuggingFace Spaces, you would need to use a supervisor or change the entrypoint. However, the easiest way for the judges is to have the UI directly accessible.

   You can update your Dockerfile to run the Gradio app on 7860, and have the Gradio app launch the FastAPI server in the background, OR keep them separate.

   **Recommended setup for HF Spaces (Single Container):**
   *Create a `run.sh` script to start both:*
   ```bash
   #!/bin/bash
   # Start FastAPI backend
   uvicorn server.app:app --host 0.0.0.0 --port 7860 &
   
   # Wait for it to start
   sleep 3
   
   # Start Gradio frontend on port 7860 (Wait, HF only exposes one port. Let's make FastAPI run on 8000 and Gradio on 7860)
   ```
