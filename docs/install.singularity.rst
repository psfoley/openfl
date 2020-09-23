.. # Copyright (C) 2020 Intel Corporation
.. # Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

.. _install_singularity:

Singularity Installation
###################

.. note::

   Make sure you've run the :ref:`the initial steps <install_initial_steps>` section first.

.. note::
    You'll need Docker installed on the node where you'll 
    be building the Singularity containers. To check
    that Docker is installed and running properly, you
    can run the Docker *Hello World* command like this:

    .. code-block:: console

      $ docker run hello-world
      Hello from Docker!
      This message shows that your installation appears to be working correctly.
      ...
      ...
      ...

.. note::
    You'll need Singularity installed on all nodes. 
    To check that Singularity is installed, run the following:

    .. code-block:: console

      $ singularity help
     
      Linux container platform optimized for High Performance Computing (HPC) and
      Enterprise Performance Computing (EPC)
      ...
      ...
      ...


1. Set the path to your Federated Learning Plan

.. code-block:: console

    $ export plan=$FLPLAN

replacing $FLPLAN with the federated plan YAML file you intend to use in your Singularity environment. For example,

.. code-block:: console

    $ export plan=keras_cnn_mnist.yaml

2.	Build the Singularity containers using the command:

.. code-block:: console

   $ make build_singularity

This will create the Docker and Singularity containers that are used by the aggregator
and the collaborators. It will append the model name and the
user that created the container. For example,
if user **abc123** ran the command with the FL Plan *keras_cnn_mnist_2.yaml*
that references the *keras_cnn* model then
the output would be:

.. code-block:: console

   $ Created Singularity container tfl_agg_keras_cnn_abc123.sif
   $ Created Singularity container tfl_col_cpu_keras_cnn_abc123.sif


