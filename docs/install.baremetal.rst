.. # Copyright (C) 2020 Intel Corporation
.. # Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

**********************
Baremetal Installation
**********************

Intel has tested the installation on Ubuntu 18.04 and Centos 7.6 systems.
A Python 3.6 virtual environment (venv) is used to isolate the packages.
The basic installation is via the Makefile included in the root directory
of the repository.

Installation Steps
##################

.. note::
   You'll need to first setup the certificates :ref:`using these instructions <install_certs>`.

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

.. note::
   You can discover the FQDN with the Linux command:

   .. code-block:: console

     $ hostname –-fqdn


4.	If pyyaml is not installed, then use pip to install it:

.. code-block:: console

   $ pip3 install pyyaml

5.	Make sure that you followed the steps in Configure the Federation and
have copied the keys and certificates onto the federation nodes.

6.	Build the virtual environment using the command:

.. code-block:: console

   $ make install

This should create a Python 3 virtual environment with the required
packages (e.g. TensorFlow, PyTorch, nibabel) that are used by
the aggregator and the collaborators. Note that you can add custom
Python packages by editing this section in the Makefile.

.. figure:: images/custom_packages.png

   How to install a custom package in the virtual environment.

Just add your own line. For example,

.. code-block:: console

   $ venv/bin/pip3 install my_package 
