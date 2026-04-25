FROM python:3.11-slim

# HuggingFace Spaces requires port 7860
ENV PORT=7860
WORKDIR /app

# Install dependencies
COPY server/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY models.py ./models.py
COPY tasks.py ./tasks.py
COPY client.py ./client.py
COPY inference.py ./inference.py
COPY server/ ./server/

# Expose port
EXPOSE 7860

# Start server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
