# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.


ARG whoami
FROM tfl_agg_pt_resnet_$whoami:0.1

# Only use one CPU thread to train the model
#   to avoid the significant communication overhead.
ENV OMP_NUM_THREADS=1
ENV KMP_BLOCKTIME=30
ENV KMP_SETTINGS=1
ENV KMP_AFFINITY=granularity=fine,verbose,compact,1,0

# FIXME: this is only for the MNIST data object and should ultimately be removed!
RUN pip3 install tensorflow==1.14.0
RUN pip3 install torch==1.3.1