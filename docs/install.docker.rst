.. # Copyright (C) 2020 Intel Corporation
.. # Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

.. _install_docker:

Docker Installation
###################

.. note::

   Make sure you've run the :ref:`Initial Installation Steps <install_initial_steps>` section first.

.. note::
    You'll need Docker installed on all nodes. To check
    that Docker is installed and running properly, you
    can run the Docker *Hello World* command like this:

    .. code-block:: console

      $ docker run hello-world
      Hello from Docker!
      This message shows that your installation appears to be working correctly.
      ...
      ...
      ...

1. Set the path to your Federated Learning Plan

.. code-block:: console

    $ export plan=$FLPLAN

replacing $FLPLAN with the federated plan YAML file you intend to use in your Docker environment. For example,

.. code-block:: console

    $ export plan=keras_cnn_mnist_2.yaml

2.	Build the Docker containers using the command:

.. code-block:: console

   $ make build_containers 

This should create the Docker containers that are used by the aggregator
and the collaborators. It will append the model name and the
user that created the container. For example,
if user **abc123** ran the command with the FL Plan *keras_cnn_mnist_2.yaml*
that references the *keras_cnn* model then
the output would be:

.. code-block:: console

   $ Successfully tagged tfl_agg_keras_cnn_abc123:0.1
   $ Successfully tagged tfl_col_cpu_keras_cnn_abc123:0.1
