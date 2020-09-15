.. # Copyright (C) 2020 Intel Corporation
.. # Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

.. _running_the_federation_singularity:

Running on Singularity
######################

We will show you how to set up |productName| on
Singularity using a simple `MNIST <https://en.wikipedia.org/wiki/MNIST_database>`_
dataset and a `PyTorch <https://www.pytorch.org/>`_
CNN model as
an example. You will note many similarities between
this tutorial and the :ref:`Docker tutorial <running_the_federation_docker>`.

Before you run the federation make sure you have followed the
instructions for the :ref:`Singularity installation <install_singularity>`.

On the Aggregator
~~~~~~~~~~~~~~~~~

1.      Follow the Singularity Installation steps as described previously.

2.      Run the Singularity container for the aggregator:

.. code-block:: console

   $ export plan=pt_cnn_low_level_api.yaml
   $ make run_agg_singularity
   [FL Singularity for Aggregator] ~/spr_secure_intelligence-trusted_federated_learning >

When the Singularity container for the aggregator begins you’ll see the prompt above.
This means you are within the running Singularity container.
You can always exit back to the original Linux shell by typing :code:`exit`.

3.      It is assumed that the federation may be fine-tuning a previously
trained model. For this reason, the pre-trained weights for the model
will be stored within protobuf files on the aggregator and passed to the
collaborators during initialization. As seen in the YAML file, the protobuf
file with the initial weights is expected to be found in the file
*pytorch_cnn_init.pbuf*. For this example, however, we’ll just create an
initial set of random model weights and putting it into that file by
running the command:

.. code-block:: console

   $ fx plan initialize -p plans/keras_cnn_mnist_2.yaml -d plans/defaults/data_tf_mnist.yaml -l plans/collaborator_lists/cols_2.yaml

.. note::

    :code:`-l cols_2.yaml` needs to be changed to the names in your collaborator list.
    A good practice is to create a new YAML file for each of your federations.
    This file is only needed by the aggregator and must be in the following format:

      .. code-block:: yaml

         collaborator_common_names :
         - 'col_0'
         - 'col_1'

4.      Now we’re ready to start the aggregator by running the fx command. During this step the
fully-qualified domain name (FQDN) for the aggregator node address
is parsed from the flplan's network configuration in order to present the correct certificate.

.. code-block:: console

   $ fx service start-agg -p plans/keras_cnn_mnist_2.yaml -l plans/collaborator_lists/cols_2.yaml

.. note::

    :code:`-l cols_2.yaml` needs to be changed to the names in your collaborator list.
    A good practice is to create a new YAML file for each of your federations.
    This file is only needed by the aggregator and must be in the following format:

      .. code-block:: yaml

         collaborator_common_names :
         - 'col_0'
         - 'col_1'

At this point, the aggregator
is running and waiting for the collaborators to connect. When all of the
collaborators connect, the aggregator starts training. When the last round
of training is complete, the aggregator stores the final weights in the
protobuf file that was specified in the YAML file
(in this case *pytorch_cnn_init.pbuf*).

On the Collaborators
~~~~~~~~~~~~~~~~~~~~

1.      Now run the Singularity on the collaborator. For example, if the collaborator
label is **col_0**, run this command:

.. code-block:: console

   $ export plan=pt_cnn_low_level_api.yaml
   $ make run_col_singularity col_name=col_0
   [FL Singularity for Collaborator col_0] ~/spr_secure_intelligence-trusted_federated_learning >


2.      Now run the collaborator fx command to start the collaborator.

.. code-block:: console

   $ fx service start-col -p plans/keras_cnn_mnist_2.yaml -n col_0 -d plans/defaults/data_tf_mnist.yaml 

3.      Repeat this for each collaborator in the federation. Once all
collaborators have joined, the aggregator will start and
you will see log messages describing the progress of the federated training.
