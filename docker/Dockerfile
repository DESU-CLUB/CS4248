FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Avoid timezone prompt during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and other dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Make python3 the default python
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Copy requirements file
COPY requirements_gpu.txt .

# Install PyTorch first with the specific index-url
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 torch>=2.6.0 torchvision>=0.21.0

# Install the rest of the packages
RUN grep -v "torch\|torchvision\|--index-url" requirements_gpu.txt > other_requirements.txt && \
    pip install --no-cache-dir -r other_requirements.txt

# Add a label for Modal
LABEL modal.com="true"

# Default command - this can be overridden by Modal
CMD ["python", "-c", "import transformers; import torch; print(f'PyTorch version: {torch.__version__}\\nTransformers version: {transformers.__version__}\\nCUDA available: {torch.cuda.is_available()}')"]
