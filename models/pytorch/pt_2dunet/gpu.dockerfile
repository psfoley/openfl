ARG whoami
FROM tfl_agg_pt_2dunet_$whoami:0.1

RUN pip3 install torch==1.2.0