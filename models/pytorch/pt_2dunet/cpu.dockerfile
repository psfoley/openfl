# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

ARG whoami
FROM tfl_agg_pt_2dunet_$whoami:0.1

# Only use one CPU thread to train the model
#   to avoid the significant communication overhead.
ENV OMP_NUM_THREADS=1
ENV KMP_BLOCKTIME=30
ENV KMP_SETTINGS=1
ENV KMP_AFFINITY=granularity=fine,verbose,compact,1,0

RUN pip3 install torch==1.2.0