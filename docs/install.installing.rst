.. # Copyright (C) 2020 Intel Corporation
.. # Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

***********************
Installing the Software
***********************

Intel has tested the installation on `Ubuntu 18.04 <https://releases.ubuntu.com/18.04/>`_
and `Centos 7.6 <https://www.centos.org/>`_ systems.
A Python 3 `virtual environment (venv) <https://docs.python-guide.org/dev/virtualenvs/#lower-level-virtualenv>`_
is used to isolate the packages.
The basic installation is via the `Makefile <https://gitlab.devtools.intel.com/secure-intelligence-team/spr_secure_intelligence-trusted_federated_learning/-/blob/master/Makefile>`_
included in the root directory
of the repository.


Initial Steps
#############

.. note::
   You'll need to first setup the certificates :ref:`using these instructions <install_certs>`.

1.	Unzip the source code

.. code-block:: console

   $ unzip spr_secure_intelligence-trusted_federated_learning.zip

2.	Change into the project directory.

.. code-block:: console

   $ cd spr_secure_intelligence-trusted_federated_learning

3. Install Python 3 and the Python `virtual environment <https://docs.python.org/3.6/library/venv.html#module-venv>`_.

.. note::
You can find the instructions on the official
`Python website <https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#installing-virtualenv>`_.
You may need to log out and back in for the changes to take effect.

   .. code-block:: console

     $ python3 -m pip install --user virtualenv


4.	Use a text editor to open the YAML file for the federation plan.

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


5.	If pyyaml is not installed, then use pip to install it:

.. code-block:: console

   $ pip3 install pyyaml

6.	Make sure that you followed the steps in :ref:`Configure the Federation <install_certs>` and
have copied the keys and certificates onto the federation nodes.

.. _install_baremetal:

Baremetal Installation
######################

.. note::

   Make sure you've run the :ref:`install.installing:Initial Steps` section first.

1.	Build the virtual environment using the command:

.. code-block:: console

   $ make install

This should create a Python 3 virtual environment with the required
packages (e.g. TensorFlow, PyTorch, OpenCV, nibabel) that are used by
the aggregator and the collaborators. Note that you can add custom
Python packages by editing this section in the Makefile.

.. figure:: images/custom_packages.png
   :scale: 80 %

   How to install a custom package in the virtual environment.

Just add your own line. For example,

.. code-block:: console

   venv/bin/pip3 install my_package 


.. _install_docker:

Docker Installation
###################

.. note::

   Make sure you've run the :ref:`install.installing:Initial Steps` section first.

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

1.	Build the Docker containers using the command:

.. code-block:: console

   $ make build_containers model_name=$DOCKER_LABEL

replacing *$DOCKER_LABEL* with whatever label you wish to give the Docker container.
For example,

.. code-block:: console

   $ make build_containers model_name=keras_cnn

This should create the Docker containers that are used by the aggregator
and the collaborators. It will append the *$DOCKER_LABEL* and the
name of the user that created the container. For example,
if user **intel123** ran the command using the Docker label *keras_cnn* then
the output would be:

.. code-block:: console

   $ Successfully tagged tfl_agg_keras_cnn_intel123:0.1
   $ Successfully tagged tfl_col_cpu_keras_cnn_intel123:0.1
