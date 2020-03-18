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

The plan we will use for this tutorial is mnist_ten_cols.yaml.


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

  $ ../venv/bin/python create_initial_weights_file_from_flplan.py -p mnist_ten_cols.yaml


4. Kick off the simulation, copying the output to a text file.

.. code-block:: console

  $ ../venv/bin/python run_simulation_from_flplan.py -p mnist_ten_cols.yaml 2>&1 | tee output_mnist_ten_cols_simulation.txt



5. Grep the output for info (one example below)

.. code-block:: console

  $ grep -A 1 round output_mnist_ten_cols_simulation.txt                                                                                                                                                                 msheller@spr-gpu01
    INFO:tfedlrn.aggregator.aggregator:round results for model id/version ConvModel/1
    INFO:tfedlrn.aggregator.aggregator:     validation: 0.8365000031888485
    --
    INFO:tfedlrn.aggregator.aggregator:round results for model id/version ConvModel/2
    INFO:tfedlrn.aggregator.aggregator:     validation: 0.9359000027179718
    --
    INFO:tfedlrn.aggregator.aggregator:round results for model id/version ConvModel/3
    INFO:tfedlrn.aggregator.aggregator:     validation: 0.9465999960899353




6. Edit the plan to train for more rounds, etc.



