[![pipeline status](https://gitlab.devtools.intel.com/weilinxu/spr_secure_intelligence-trusted_federated_learning/badges/master/pipeline.svg)](https://gitlab.devtools.intel.com/weilinxu/spr_secure_intelligence-trusted_federated_learning/commits/master)
[![coverage report](https://gitlab.devtools.intel.com/weilinxu/spr_secure_intelligence-trusted_federated_learning/badges/master/coverage.svg)](https://gitlab.devtools.intel.com/weilinxu/spr_secure_intelligence-trusted_federated_learning/commits/master)

# Welcome to Intel® Federated Learning

[Federated learning](https://en.wikipedia.org/wiki/Federated_learning) is a distributed machine learning approach that
enables organizations to collaborate on machine learning projects
without sharing sensitive data, such as, patient records, financial data,
or classified secrets ([McMahan, 2016](https://arxiv.org/abs/1602.05629);
[Sheller, Reina, Edwards, Martin, & Bakas, 2019](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6589345);
[Yang, Liu, Chen, & Tong, 2019](https://arxiv.org/abs/1902.04885); 
[Sheller et al., 2020](https://www.nature.com/articles/s41598-020-69250-1)).
The basic premise behind federated learning
is that the model moves to meet the data rather than the data moving
to meet the model. Therefore, the minimum data movement needed
across the federation is solely the model parameters and their updates.


Intel® Federated Learning is a Python 3 project developed by Intel Labs and 
Intel Internet of Things Group. It is released under the [Apache License, Version 2](https://www.apache.org/licenses/LICENSE-2.0).

![Federated Learning](docs/images/diagram_fl.png)

## Requirements

- OS: Tested on Ubuntu Linux 16.04 and 18.04.
- Python 3.6+ with a Python virtual environment (e.g. [conda](https://docs.conda.io/en/latest/))
- TensorFlow 2+ or PyTorch 1.6+ (depending on your training requirements). Other frameworks can be supported.



### OpenFL and the Federated Tumor Segmentation (FeTS) Initiative

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
built to be agnostic to the use-case, the industry, and the 
machine learning framework.

