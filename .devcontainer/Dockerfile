# Use the official TensorFlow GPU image as the base
FROM tensorflow/tensorflow:2.16.1-gpu-jupyter

# Install additional utilities, including Node.js and npm
RUN apt-get update && apt-get install -y \
    git \
    curl \
    vim \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /workspaces/cancer-project