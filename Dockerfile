FROM python:3.10-slim

# Install system dependencies required for FAISS and compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the optimized requirements file
COPY requirements-horus.txt .

# 1. Install PyTorch CPU-only wheel to avoid downloading ~2GB of CUDA runtimes
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 2. Install the rest of the application dependencies
RUN pip install --no-cache-dir -r requirements-horus.txt

# Copy the horus application package code
COPY horus/ /app/horus/

# Expose Streamlit default port
EXPOSE 8501

# Run streamlit
CMD ["streamlit", "run", "horus/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
