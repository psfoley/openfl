# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

# docker build --build-arg HTTPS_PROXY=$HTTPS_PROXY --build-arg HTTP_PROXY=$HTTP_PROXY -t fledge .

# docker build . -t fledge

ARG BASE_IMAGE=intel/intel-optimized-tensorflow
FROM $BASE_IMAGE

LABEL maintainer "Weilin Xu <weilin.xu@intel.com>"

ENV install_dir=/usr/fledge/

WORKDIR /home/fledge

ADD . $install_dir

# Install the fledge package and its dependency.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-venv
RUN pip install --upgrade pip
RUN pip install $install_dir

