# Welcome to Intel Federated Learning

[![pipeline status](https://gitlab.devtools.intel.com/weilinxu/spr_secure_intelligence-trusted_federated_learning/badges/master/pipeline.svg)](https://gitlab.devtools.intel.com/weilinxu/spr_secure_intelligence-trusted_federated_learning/commits/master)
[![coverage report](https://gitlab.devtools.intel.com/weilinxu/spr_secure_intelligence-trusted_federated_learning/badges/master/coverage.svg)](https://gitlab.devtools.intel.com/weilinxu/spr_secure_intelligence-trusted_federated_learning/commits/master)

Intel\ :sup:`|reg|` Federated Learning is a Python 3 library for 
[federated learning](https://en.wikipedia.org/wiki/Federated_learning). 
It enables organizations to collaborately train a model without 
sharing sensitive information.

There are basically two components in the library: the collaborator which 
uses local dataset to train a global model 
and the aggregator which receives model updates from 
collaborators and combines them to form the global model.

The aggregator is framework-agnostic, while the collaborator can use any 
deep learning frameworks, such as TensorFlow or PyTorch.

Intel\ :sup:`|reg|` Federated Learning is developed by Intel Labs and 
Intel Internet of Things Group.


## OpenFL and the Federated Tumor Segmentation (FeTS) Initiative

This project extends the [Open Federated Learning (OpenFL)](https://github.com/IntelLabs/OpenFederatedLearning) framework that was 
developed as part of a collaboration between Intel 
and the University of Pennsylvania (UPenn) for federated learning. 
It describes Intel’s commitment in 
supporting the grant awarded to the [Center for Biomedical Image Computing and Analytics (CBICA)](https://www.cbica.upenn.edu/) 
at UPenn (PI: S.Bakas) from the [Informatics Technology for Cancer Research (ITCR)](https://itcr.cancer.gov/) program of 
the National Cancer Institute (NCI) of the National Institutes of Health (NIH), 
for the development of the [Federated Tumor Segmentation (FeTS, www.fets.ai)](https://www.fets.ai/) 
platform (grant award number: U01-CA242871). 

FeTS is an exciting, real-world 
medical FL platform, and we are honored to be collaborating with UPenn in 
leading a federation of international collaborators. OpenFL was 
designed to serve as the backend for the FeTS platform, and OpenFL developers 
and researchers continue to work very closely with UPenn on 
the FeTS project.

We’ve included the [FeTS-AI/Algorithms](https://github.com/FETS-AI/Algorithms) 
repository as a submodule of OpenFL to highlight how OpenFL serves as the FeTS 
backend. While not necessary to run the framework, the FeTS algorithms show 
real-world FL models and use cases. Additionally, the 
[FeTS-AI/Front-End](https://github.com/FETS-AI/Front-End) shows how UPenn 
and Intel have integrated UPenn’s medical AI expertise with Intel’s framework 
to create a federated learning solution for medical imaging. 

Although initially developed for use in medical imaging, this project was 
built to be agnostic to the use-case and the 
machine learning framework, and we welcome input from domains 
outside medicine and imaging.



### Requirements

- OS: Primarily tested on Ubuntu 16.04 and 18.04, but code should be OS-agnostic. (Optional shell scripts may not be).
- Python 3.6+ with a Python virtual environment (e.g. [conda](https://docs.conda.io/en/latest/))
- TensorFlow 2+ or PyTorch 1.6+ (depending on your training requirements-- other frameworks can be supported)


