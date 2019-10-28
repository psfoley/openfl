[![pipeline status](https://gitlab.devtools.intel.com/weilinxu/spr_secure_intelligence-trusted_federated_learning/badges/master/pipeline.svg)](https://gitlab.devtools.intel.com/weilinxu/spr_secure_intelligence-trusted_federated_learning/commits/master)
[![coverage report](https://gitlab.devtools.intel.com/weilinxu/spr_secure_intelligence-trusted_federated_learning/badges/master/coverage.svg)](https://gitlab.devtools.intel.com/weilinxu/spr_secure_intelligence-trusted_federated_learning/commits/master)



Installation
------------

I'm still not sure the ideal way to do installation. Make and pipenv have some annoying interactions I need to solve (particularly, I would like to use make to determine if the venv is installed, but pipenv venv names have id-strings attached, so I'm not sure what the venv path will be).

For now, here is the method to build and install as a user (not developer):

1. run pipenv install in the root folder of the project to create the new virtual env with setuptools and wheel updated.
2. run python setup.py bdist_wheel to create the wheel
3. run pip install 