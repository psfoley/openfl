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
                'models.tensorflow',
                'models.keras', 
                'data', 
                'data.pytorch', 
                'data.tensorflow', 
                'data.keras'],
      install_requires=['tensorflow==1.14.0', 'torch==1.1.0', 'protobuf', 'pyyaml', 'grpcio', 'tqdm', 'coloredlogs', 'tensorboardX', 'nibabel']
)
