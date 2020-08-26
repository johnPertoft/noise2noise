FROM tensorflow/tensorflow:1.15.2-gpu-py3

RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1

RUN pip install opencv-python==3.4.5.20

RUN useradd -ms /bin/bash n2n
USER n2n
WORKDIR /home/n2n
