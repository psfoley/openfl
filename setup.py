from setuptools import setup

setup(name='tfedlrn',
      version='0.0.0',
      package_dir={'tfedlrn': 'src/tfedlrn'},
      packages=['tfedlrn', 'tfedlrn.aggregator', 'tfedlrn.collaborator', 'tfedlrn.proto', 'tfedlrn.collaborator.pytorchmodels'],
      # 'tfedlrn.bratsunettest', 
#      install_requires=['tensorflow', 'tqdm', 'scipy'])
      )
