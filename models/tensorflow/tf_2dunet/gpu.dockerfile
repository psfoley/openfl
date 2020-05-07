# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

ARG whoami
FROM tfl_agg_tf_2dunet_$whoami:0.1

RUN pip3 install tensorflow-gpu==1.14.0;