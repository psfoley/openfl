# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.


ARG whoami
FROM tfl_agg_pt_resnet_$whoami:0.1

# FIXME: this is only for the MNIST data object and should ultimately be removed!
RUN pip3 install tensorflow-gpu==1.14.0
RUN pip3 install torch==1.3.1