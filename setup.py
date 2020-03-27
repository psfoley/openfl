# Copyright (C) 2020 Intel Corporation

from setuptools import setup

setup(name='tfedlrn',
      version='0.0.0',
      packages=['tfedlrn',
                'tfedlrn.aggregator',
                'tfedlrn.collaborator',
                'tfedlrn.proto',
                'models',
                'models.pytorch', 
                'models.pytorch.pt_2dunet', 
                'models.pytorch.pt_cnn',
                'models.tensorflow', 
                'models.tensorflow.keras_cnn', 
                'models.tensorflow.tf_2dunet',
                'data', 
                'data.pytorch', 
                'data.tensorflow'],
      install_requires=['tensorflow==1.14.0', 'torch==1.3.1', 'protobuf', 'pyyaml', 'grpcio', 'tqdm', 'coloredlogs', 'tensorboardX', 'nibabel']
)
