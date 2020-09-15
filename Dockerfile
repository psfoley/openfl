# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

# docker build --build-arg HTTPS_PROXY=$HTTPS_PROXY --build-arg HTTP_PROXY=$HTTP_PROXY -t fledge .

ARG BASE_IMAGE=intel/intel-optimized-tensorflow
FROM $BASE_IMAGE

LABEL maintainer "Weilin Xu <weilin.xu@intel.com>"

WORKDIR /home/fledge

# Set up the proxy servers before building the Docker image.
# Credit goes to Cory Cornelius.
# ARG http_proxy
# ARG HTTP_PROXY
# ARG HTTPS_PROXY
# ARG https_proxy
# ARG socks_proxy
# ARG ftp_proxy
# ARG no_proxy

# ENV http_proxy=$http_proxy
# ENV https_proxy=$https_proxy
# ENV HTTP_PROXY=$HTTP_PROXY
# ENV HTTPS_PROXY=$HTTPS_PROXY
# ENV socks_proxy=$socks_proxy
# ENV ftp_proxy=$ftp_proxy
# ENV no_proxy=$no_proxy

# RUN apt-get update && apt-get install -y \
#   python3-pip \
#   python3-venv
# RUN pip3 install --upgrade pip

# Create a user with the same UID so that it is easier to access the mapped volume.
# ARG UNAME=fledge
# ARG UID=1000
# ARG GID=1000

# RUN groupadd -g $GID $UNAME; exit 0
# RUN useradd --no-log-init -m -u $UID -g $GID -s /bin/bash $UNAME
# USER $UNAME

ADD . $WORKDIR

# Install the fledge package and its dependency.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-venv
RUN pip install --upgrade pip
RUN pip install -e .
