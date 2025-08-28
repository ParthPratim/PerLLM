FROM nvcr.io/nvidia/pytorch:25.08-py3

# CORE Libraries for ML-DL
RUN pip install transformers datasets accelerate 
RUN pip install pandas matplotlib scikit-learn scipy
RUN pip install joblib dill
RUN pip install tokenizers 
RUN pip install trl peft 
RUN pip install deepspeed
RUN pip install vllm 
RUN pip install bitsandbytes
RUN pip install llama-index
RUN pip install faiss


# Distributed computing and paralellism
RUN pip install triton jax
RUN pip install ray dask 

# Install C++ toolchain and build essentials for CUDA Programming
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install common CUDA dev libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnccl-dev \
    libcusparse-dev \
    libcusolver-dev \
    libcublas-dev \
    libnvtoolsext-dev \
    libcurand-dev \
    libnpp-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Boost (headers + system libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

# Install OpenMPI for distributed CUDA programming
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenmpi-dev openmpi-bin \
    && rm -rf /var/lib/apt/lists/*