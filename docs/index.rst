.. # Copyright (C) 2020 Intel Corporation
.. # Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.


.. Documentation master file, created by
   sphinx-quickstart on Thu Oct 24 15:07:19 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

***************************************
Welcome to |productName| documentation!
***************************************

|productName| is a Python3 library for federated learning.
It enables organizations to collaborately train a
model without sharing sensitive information with each other.

There are basically two components in the library:
the *collaborator* which uses local sensitive dataset to fine-tune
the aggregated model and the *aggregator* which receives
model updates from collaborators and distribute the aggregated
models.

The *aggregator* is framework-anostic, while the *collaborator*
can use any deep learning frameworks, such as Tensorflow or
Pytorch.

|productName| is developed by Intel Labs and Intel Internet of Things Group.

test

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   .. toctree::

      overview
      .. toctree::

         install.design
         install.certificates
         .. toctree::

            install.initial
            install.baremetal


      running_the_federation

   tfedlrn
   models


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
