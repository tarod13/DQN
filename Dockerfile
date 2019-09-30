FROM ubuntu:18.04

WORKDIR /home/dg/Research/docker

RUN apt-get update && apt-get install -y apt-utils && apt-get install -y curl
RUN apt-get install -y python3.6
RUN apt-get install -y python3-pip
RUN ln -s /usr/bin/python3.6 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip
RUN apt-get -y install git
RUN pip install --upgrade pip

RUN pip install torch==1.2.0+cpu torchvision==0.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install -U pip setuptools
RUN apt-get install -y libsm6 libxrender-dev
RUN pip install opencv-python
RUN pip install cython
RUN apt-get install zlib1g-dev

RUN pip install tensorflow==2.0.0-rc1
RUN pip install gym
RUN pip install gym[atari]
RUN python -m pip install -U matplotlib
