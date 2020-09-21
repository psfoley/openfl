.. # Copyright (C) 2020 Intel Corporation
.. # Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

.. _running_the_federation_docker:

Running on Docker
#################

We will show you how to set up |productName| on
Docker using a simple `MNIST <https://en.wikipedia.org/wiki/MNIST_database>`_
dataset and a `TensorFlow/Keras <https://www.tensorflow.org/>`_
CNN model as
an example. You will note that this is literally the
same code as the :ref:`Baremetal <running_baremetal>`, but we are simply wrapping
the venv within a Docker container.

Before you run the federation make sure you have followed the
instructions for the :ref:`Docker installation <install_docker>`.

On the Aggregator
~~~~~~~~~~~~~~~~~

1.	Follow the Docker Installation steps as described previously.

2.	Run the Docker container for the aggregator:

.. code-block:: console

   $ export plan=keras_cnn_mnist_2.yaml
   $ make run_agg_container

When the Docker container for the aggregator begins you’ll see the prompt above.
This means you are within the running Docker container.
You can always exit back to the original Linux shell by typing :code:`exit`.

3.	It is assumed that the federation may be fine-tuning a previously
trained model. For this reason, the pre-trained weights for the model
will be stored within protobuf files on the aggregator and passed to the
collaborators during initialization. As seen in the YAML file, the protobuf
file with the initial weights is expected to be found in the file
*keras_cnn_mnist_init.pbuf*. For this example, however, we’ll just create an
initial set of random model weights and putting it into that file by
running the command:

.. code-block:: console

   $ fx plan initialize -p plan.yaml -d data.yaml  -c cols.yaml

.. note::

    :code:`-c cols.yaml` needs to be changed to the names in your collaborator list.
    A good practice is to create a new YAML file for each of your federations.
    This file is only needed by the aggregator and must be in the following format:

      .. code-block:: yaml

         collaborators:
         - 'one'
         - 'two'

At this point, the aggregator
is running and waiting for the collaborators to connect. When all of the
collaborators connect, the aggregator starts training. When the last round
of training is complete, the aggregator stores the final weights in the
protobuf file that was specified in the YAML file
(in this case *keras_cnn_mnist_latest.pbuf*).

On the Collaborators
~~~~~~~~~~~~~~~~~~~~

1.	Now run the Docker on the collaborator. For example, if the collaborator
label is **one**, run this command:


2.	Now run the collaborator fx command to start the collaborator.

3.	Repeat this for each collaborator in the federation. Once all
collaborators have joined, the aggregator will start and
you will see log messages describing the progress of the federated training.
