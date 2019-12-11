from setuptools import setup

setup(name='tfedlrn',
      version='0.0.0',
      package_dir={'tfedlrn': 'tfedlrn'},
      packages=['tfedlrn', 'tfedlrn.aggregator', 'tfedlrn.collaborator', 'tfedlrn.coordinator', 'tfedlrn.proto'],
      # 'tfedlrn.bratsunettest', 
      install_requires=['protobuf', 'coloredlogs', 'pyyaml', 'nibabel', 'tensorboardX', 'grpcio']
)
