.. # Copyright (C) 2020 Intel Corporation
.. # Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

.. _tutorial-new-task:

How to add a new task
-----------------------

We will learn how to add a new task of training a 2D-UNet model
on the BraTS dataset in the federated learning fashion.

1. Model development
^^^^^^^^^^^^^^^^^^^^^
First off, we need a piece of software that any collaborator can
use to train a model with a local dataset.

See :ref:`tutorial-new-model` to learn how to adapt model training code
so that it can be used in our federated learning framework.


2. Dataset preparation
^^^^^^^^^^^^^^^^^^^^^^^
Assume we are one of the collaborators in federated
learning, the first thing we need to do is prepareing
a local dataset that can be consumed by the model training
code.


3. 
