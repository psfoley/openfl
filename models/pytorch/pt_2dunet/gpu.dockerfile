ARG whoami
FROM tfl_agg_unet_2d_pt_$whoami:0.1

RUN pip3 install torch==1.1.0