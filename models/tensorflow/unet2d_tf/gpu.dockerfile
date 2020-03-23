# Copyright (C) 2020 Intel Corporation
# Licensed subject to Collaboration Agreement dated February 28th, 2020 between Intel Corporation and Trustees of the University of Pennsylvania.

ARG whoami
FROM tfl_agg_unet2d_tf_$whoami:0.1

RUN pip3 install tensorflow-gpu==1.14.0;