FROM tfl_agg:0.1

ENV OMP_NUM_THREADS=112
ENV KMP_BLOCKTIME=30
ENV KMP_SETTINGS=1
ENV KMP_AFFINITY=granularity=fine,verbose,compact,1,0

RUN ./venv/bin/pip3 install intel-tensorflow==1.14.0;