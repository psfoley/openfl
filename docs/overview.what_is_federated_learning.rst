.. # Copyright (C) 2020 Intel Corporation
.. # Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

***************************
What is Federated Learning?
***************************

`Federated learning <https://en.wikipedia.org/wiki/Federated_learning>`_ is a distributed machine learning approach that
enables organizations to collaborate on machine learning projects
without sharing sensitive data, such as, patient records, financial data,
or classified secrets (`McMahan, 2016 <https://arxiv.org/abs/1602.05629>`_;
`Sheller, Reina, Edwards, Martin, & Bakas, 2019 <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6589345/>`_;
`Yang, Liu, Chen, & Tong, 2019 <https://arxiv.org/abs/1902.04885>`_).
The basic premise behind federated learning
is that the model moves to meet the data rather than the data moving
to meet the model. Therefore, the minimum data movement needed
across the federation is solely the model parameters and their updates.

.. figure:: images/diagram_fl.png

   Federated Learning
