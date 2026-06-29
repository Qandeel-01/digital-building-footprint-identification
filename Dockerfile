FROM ubuntu:18.04

# System setup with Python 3.6
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.6 \
        python3-pip \
        python3.6-dev \
        gdal-bin \
        libgdal-dev \
        && \
    ln -s /usr/bin/python3.6 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install GDAL with explicit version (no awk parsing)
RUN GDAL_VERSION=$(gdal-config --version) && \
    pip install GDAL==${GDAL_VERSION}

# Install essential packages
RUN pip install --upgrade pip wheel setuptools

# Copy only necessary directories
WORKDIR /app
COPY neat_eo ./neat_eo
COPY ada_tools ./ada_tools

# Install project-specific packages
RUN pip install ./neat_eo ./ada_tools && \
    pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

WORKDIR /workspace