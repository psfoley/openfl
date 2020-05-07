# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

ARG whoami
FROM tfl_agg_pt_2dunet_$whoami:0.1

RUN pip3 install torch==1.2.0