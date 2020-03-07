ARG whoami
FROM tfl_agg_unet2d_tf_$whoami:0.1

RUN pip3 install tensorflow-gpu==1.14.0;