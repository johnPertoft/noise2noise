FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1

RUN add-apt-repository ppa:jonathonf/python-3.6 -y
RUN apt-get update && apt-get install -y \
    python3.6 \
    python3.6-dev \
    python3.6-venv
RUN wget https://bootstrap.pypa.io/get-pip.py && python3.6 get-pip.py
RUN ln -s /usr/bin/python3.6 /usr/local/bin/python

RUN pip3.6 install \
    tensorflow-gpu==1.12 \
    numpy==1.15.4 \
    opencv-python==3.4.5

RUN useradd -ms /bin/bash n2n
USER n2n
WORKDIR /home/n2n