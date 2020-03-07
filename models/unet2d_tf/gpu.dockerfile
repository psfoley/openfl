ARG whoami
FROM tfl_agg_UNet2D_TF_$whoami:0.1

RUN pip3 install tensorflow-gpu==1.14.0;