FROM nvidia/cuda:11.0-cudnn8-devel-ubuntu18.04

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    python3-wheel \
    python3-setuptools && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN pip3 install --no-cache-dir -U install setuptools pip
RUN pip3 install --no-cache-dir cupy-cuda110==8.0.0 scipy optuna
RUN pip3 install --no-cache-dir jupyter matplotlib pillow tqdm

CMD ["bash", "-c", "jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root"]

# docker run -it --rm --gpus all -p 8888:8888 lfd-prefiltering