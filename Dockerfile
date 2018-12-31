FROM nvidia/cuda:9.2-runtime-ubuntu18.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip3 install \
    tensorflow-gpu \
    numpy \
    opencv-python

RUN useradd -ms /bin/bash n2n
USER n2n