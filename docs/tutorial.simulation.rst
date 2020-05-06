.. # Copyright (C) 2020 Intel Corporation
.. # Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

How to run federated learning simulations (no network, single process)
-------------------------------------------

Note that much of the code used for simulation (ex. collaborator and aggregator objects) is the
same as for the multiprocess solution with grpc. Since the collaborator calls aggregator object 
methods via the grpc channel object, simulation is performed by simply replacing the channel object
provided to each collaborator with the aggregator object.

Muti-process federations as well as simulations are run from an flplan. Current flplans can be found in 
spr_secure_intelligence-trusted_federated_learning/bin/federations/plans. 

**Note that "spr_secure_intelligence-trusted_federated_learning"
is the folder name we chose for the local repository.
It can be changed to anything of your choice on your machine.**

The plan we will use for this tutorial is keras_cnn_mnist_10.yaml.


Simulated Federated Training of an MNIST Classifier across 10 Collaborators
-------------------------------------------

Create the project virtual environment
^^^^^^^^^^^^^^^^^^^^^^^^

1. To prepare, make sure you have python 3.5 (or higher) with virtualenvs installed. 


2. Enter the project folder, create the virtual environment, 
and cd to the bin directory.


.. code-block:: console

  $ cd spr_secure_intelligence-trusted_federated_learning
  $ make clean
  $ make install
  $ cd bin

3. Create the initial weights file for the model to be trained.

.. code-block:: console

  $ ../venv/bin/python create_initial_weights_file_from_flplan.py -p keras_cnn_mnist_10.yaml


4. Kick off the simulation.

.. code-block:: console

  $ ../venv/bin/python run_simulation_from_flplan.py -p keras_cnn_mnist_10.yaml



5. You'll find the output from the aggregator in bin/logs/aggregator.log. Grep this file to see results (one example below). You can check the progress as the simulation runs, if desired.

.. code-block:: console

  $ pwd                                                                                                                                                                                                                            msheller@spr-gpu01
    /home/<user>/git/tfedlrn/bin
  $ grep -A 2 "round results" logs/aggregator.log
    2020-03-30 13:45:33,404 - tfedlrn.aggregator.aggregator - INFO - round results for model id/version KerasCNN/1
    2020-03-30 13:45:33,404 - tfedlrn.aggregator.aggregator - INFO -        validation: 0.4465000107884407
    2020-03-30 13:45:33,404 - tfedlrn.aggregator.aggregator - INFO -        loss: 1.0632034242153168
    --
    2020-03-30 13:45:35,127 - tfedlrn.aggregator.aggregator - INFO - round results for model id/version KerasCNN/2
    2020-03-30 13:45:35,127 - tfedlrn.aggregator.aggregator - INFO -        validation: 0.8630000054836273
    2020-03-30 13:45:35,127 - tfedlrn.aggregator.aggregator - INFO -        loss: 0.41314733028411865
    --

Note that aggregator.log is always appended to, so will include results from previous runs.

6. Edit the plan to train for more rounds, etc.



