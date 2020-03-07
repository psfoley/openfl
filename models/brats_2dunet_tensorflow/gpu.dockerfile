ARG whoami
FROM tfl_agg_brats_2dunet_tensorflow_$whoami:0.1

RUN pip3 install tensorflow-gpu==1.14.0;