.. # Copyright (C) 2020 Intel Corporation
.. # Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.


Docker Installation
===================

We will show you how to set up Intel\ :sup:`®` \ Federated Learning on
Docker using a simple MNIST dataset and a TensorFlow/Keras CNN model as
an example. You will note that this is literally the same code as the
:ref:`Baremetal Installation <install_baremetal>`, but we are simply wrapping
the venv within a Docker container.
Before we start the tutorial, please make sure you have Docker
installed and configured properly. Here is an easy test to run in
order to test some basic functionality:

.. code-block:: console

  $ docker run hello-world
  Hello from Docker!
  This message shows that your installation appears to be working correctly.
  ...
  ...
  ...

### Installation Steps

NOTE: You'll need to first setup the certificates
:ref:`using these instructions <_install_certs>`.

1.	Unzip the source code

.. code-block:: console

   $ unzip spr_secure_intelligence-trusted_federated_learning.zip

2.	Change into the project directory.

.. code-block:: console

   $ cd spr_secure_intelligence-trusted_federated_learning

3.	Use a text editor to open the YAML file for the federation plan.

.. code-block:: console

   $ vi bin/federations/plans/keras_cnn_mnist_2.yaml

This YAML file defines the IP addresses for the aggregator. It is the main
file that controls all of the execution of the federation.
By default, the YAML file is defining a federation where the aggregator
runs on the localhost at port 5050 (it is up to the developer
to make sure that the port chosen is open and accessible to all participants).
For this demo, we’ll just focus on running everything on the same server.
You’ll need to edit the YAML file and replace localhost with the
aggregator address. Please make sure you specify the fully-qualified
domain name (`FQDN <https://en.wikipedia.org/wiki/Fully_qualified_domain_name>`_)
address (required for security). For example:

You can discover the FQDN by running the Linux command:

.. code-block:: console

   $ hostname --fqdn

4.	If pyyaml is not installed, then use pip to install it:

.. code-block:: console

   $ pip3 install pyyaml

5.	Make sure that you followed the steps in Configure the Federation and
have copied the keys and certificates onto the federation nodes.

6.	Build the Docker containers using the command:

.. code-block:: console

   $ make build_containers model_name=DOCKER_LABEL

replacing *DOCKER_LABEL* with whatever label you wish to give the Docker container.

This should create the Docker containers that are used by the aggregator
and the collaborators. It will append the DOCKER_LABEL and the
name of the user that created the container.

.. code-block:: console

   Successfully tagged tfl_agg_DOCKER_LABEL_USERNAME:0.1
   Successfully tagged tfl_col_cpu_DOCKER_LABEL_USERNAME:0.1

 
