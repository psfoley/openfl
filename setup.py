from setuptools import setup

setup(name='tfedlrn',
      version='0.0.0',
      package_dir={'tfedlrn': 'tfedlrn'},
      packages=['tfedlrn', 'tfedlrn.aggregator', 'tfedlrn.collaborator', 'tfedlrn.proto', 'tfedlrn.collaborator.tensorflowmodels', 'tfedlrn.collaborator.pytorchmodels'],
      # 'tfedlrn.bratsunettest', 
      install_requires=['tensorflow-gpu==1.10.0', 'torch', 'protobuf', 'pyzmq']
)