.. # Copyright (C) 2020 Intel Corporation
.. # Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

.. _running_baremetal:

Running on Baremetal
####################

We will show you how to set up |productName| using a simple `MNIST <https://en.wikipedia.org/wiki/MNIST_database>`_
dataset and a `TensorFlow/Keras <https://www.tensorflow.org/>`_
CNN model as an example.

Before you run the federation make sure you have followed the
instructions for the :ref:`Baremetal installation <install_baremetal>`.

On the Aggregator
~~~~~~~~~~~~~~~~~

1.	It is assumed that the federation may be fine-tuning a previously
trained model. For this reason, the pre-trained weights for the model
will be stored within protobuf files on the aggregator and
passed to the collaborators during initialization. As seen in
the YAML file, the protobuf file with the initial weights is
expected to be found in the file keras_cnn_mnist_init.pbuf. For
this example, however, we’ll just create an initial set of
random model weights and putting it into that file by running the command:

.. code-block:: console

   $ ./venv/bin/python3 ./bin/create_initial_weights_file_from_flplan.py -p keras_cnn_mnist_2.yaml -dc local_data_config.yaml --collaborators_file cols_2.yaml

.. note::

    :code:`--collaborators_file cols_2.yaml` needs to be changed to the names in your collaborator list.
    A good practice is to create a new YAML file for each of your federations. This file is only needed by the aggregator.
    These YAML files can be found in :code:`bin/federations/collaborator_lists/`


2.	Now we’re ready to start the aggregator by running the Python script. Note that we will need to pass in the fully-qualified domain name (FQDN) for the aggregator node address in order to present the correct certificate.

.. code-block:: console

   $ ./venv/bin/python3 ./bin/run_aggregator_from_flplan.py -p keras_cnn_mnist_2.yaml -ccn AGGREGATOR.FULLY.QUALIFIED.DOMAIN.NAME --collaborators_file cols_2.yaml

.. note::

    :code:`--collaborators_file cols_2.yaml` needs to be changed to the names in your collaborator list.
    A good practice is to create a new YAML file for each of your federations. This file is only needed by the aggregator.
    These YAML files can be found in :code:`bin/federations/collaborator_lists/`

At this point, the aggregator is running and waiting
for the collaborators to connect. When all of the collaborators
connect, the aggregator starts training. When the last round of
training is complete, the aggregator stores the final weights in
the protobuf file that was specified in the YAML file
(in this case *keras_cnn_mnist_latest.pbuf*).

On the Collaborator
~~~~~~~~~~~~~~~~~~~

1.	Make sure that you followed the steps in :ref:`Configure the Federation <install_certs>` and have copied the keys and certificates onto the federation nodes.

2.	Copy the plan file (e.g. *keras_cnn_mnist_2.yaml*) from the aggregator
over to the collaborator to the plan subdirectory (**bin/federations/plans**)

3.	Build the virtual environment using the command:

.. code-block:: console

   $ make install

4.	Now run the collaborator col_1 using the Python script. Again,
you will need to pass in the fully qualified domain name in
order to present the correct certificate.

.. code-block:: console

   $ ./venv/bin/python3 ./bin/run_collaborator_from_flplan.py -p keras_cnn_mnist_2.yaml -col col_1 -ccn COLLABORATOR.FULLY.QUALIFIED.DOMAIN.NAME

5.	Repeat this for each collaborator in the federation. Once all
collaborators have joined, the aggregator will start and you
will see log messages describing the progress of the federated training.
