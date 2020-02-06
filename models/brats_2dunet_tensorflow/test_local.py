#!/usr/bin/env python
"""Test the model locally before we run federated learning.

TODO: We need a complete testing for the FL model APIs.
"""
import os
os.environ["OMP_NUM_THREADS"] = "112"
os.environ["KMP_BLOCKTIME"] = "30"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"

import sys
sys.path.append('../../')

import tensorflow as tf
from timeit import default_timer as timer
from models.brats_2dunet_tensorflow import get_model

seed = 1234
model = get_model('/raid/datasets/BraTS17/MICCAI_BraTS17_Data_Training/HGG/', percent_train=0.8, shuffle=True, batch_size=1024)

val_results = model.validate()
print("Initial val_results", val_results)

epochs = 16
for e in range(epochs):
    start = timer()
    ret = model.train_epoch()
    end = timer()
    print("Time for train_epoch(): {} sec.".format(round(end - start, 3)))
    print("training results", round(ret, 4))

    start = timer()
    val_results = model.validate()
    end = timer()
    print("Time for validate(): %.1f sec." % (end - start))
    print("Epoch {} val_results".format(e), round(val_results, 4))
