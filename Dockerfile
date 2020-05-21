# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

ARG BASE_IMAGE=ubuntu:18.04
FROM $BASE_IMAGE

LABEL maintainer "Weilin Xu <weilin.xu@intel.com>"

# Set up the proxy servers before building the Docker image.
# Credit goes to Cory Cornelius. 
ARG http_proxy
ARG https_proxy
ARG socks_proxy
ARG ftp_proxy
ARG no_proxy

ENV http_proxy=$http_proxy
ENV https_proxy=$https_proxy
ENV socks_proxy=$socks_proxy
ENV ftp_proxy=$ftp_proxy
ENV no_proxy=$no_proxy

RUN apt-get update && apt-get install -y \
  python3-pip \
  python3-venv
RUN pip3 install --upgrade pip

# Create a user with the same UID so that it is easier to access the mapped volume.
ARG UNAME=testuser
ARG UID=1000
ARG GID=1000

RUN groupadd -g $GID $UNAME; exit 0
RUN useradd --no-log-init -m -u $UID -g $GID -s /bin/bash $UNAME
USER $UNAME

# Copy the artifacts.
RUN mkdir /home/${UNAME}/tfl
WORKDIR /home/${UNAME}/tfl
# FIXME: The COPY --chown does not recognize ENV or ARG yet.
USER root
COPY . .
RUN chown -R ${UNAME}:${UNAME} *
USER $UNAME

# Install the tfedlrn package and its dependency.
RUN make install
ENV PATH=/home/${UNAME}/tfl/venv/bin:$PATH
