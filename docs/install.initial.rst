.. # Copyright (C) 2020 Intel Corporation
.. # Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

Initial Steps
#############

.. note::
   You'll need to first setup the certificates :ref:`using these instructions <install_certs>`.

1.	Unzip the source code |productZip|

.. code-block:: console
   :substitutions:

   $ unzip |productZip|

2.	Change into the project directory |productDir|.

.. code-block:: console
   :substitutions:

   $ cd |productDir|

3. Install Python 3 and the Python `virtual environment <https://docs.python.org/3.6/library/venv.html#module-venv>`_.

.. note::
You can find the instructions on the official
`Python website <https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#installing-virtualenv>`_.
You may need to log out and back in for the changes to take effect.

.. code-block:: console

   $ python3 -m pip install --user virtualenv


.. note::
   If you have trouble installing the virtual environment, make sure you have Python 3 installed on your OS. For example, on Ubuntu:

   .. code-block:: console

     $ sudo apt-get install python3-pip


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

     $ hostname --fqdn-all | awk '{print $1}'


5.	If pyyaml is not installed, then use pip to install it:

.. code-block:: console

   $ pip3 install pyyaml

6.	Make sure that you followed the steps in :ref:`Configure the Federation <install_certs>` and
have copied the keys and certificates onto the federation nodes.
