.. # Copyright (C) 2020 Intel Corporation
.. # Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.


Configure The Federation
========================

TLS encryption is used for the network connections. Therefore, security keys and certificates will need to be created for the aggregator and collaborators to negotiate the connection securely. For the “Hello Federation” demo we will run the aggregator and collaborators on the same localhost server so these configuration steps just need to be done once on that machine.

Steps:

All Nodes

1.	 Unzip the source code

.. code-block:: console

  $ unzip OpenFederatedLearning-master.zip

2.	Change into the OpenFederatedLearning-master subdirectory.

.. code-block:: console

  $ cd OpenFederatedLearning-master

On the Aggregator Node

1.	Change the directory to bin/federations/pki:

.. code-block:: console

  $ cd bin/federations/pki

2.	Run the Certificate Authority script. This will setup the Aggregator node as the Certificate Authority for the Federation. All certificates will be signed by the aggregator. Follow the command-line instructions and enter in the information as prompted. The script will create a simple database file to keep track of all issued certificates.

.. code-block:: console

  $ bash setup_ca.sh

3.	Run the aggregator cert script, replacing AGGREGATOR.FULLY.QUALIFIED.DOMAIN.NAME with the actual FQDN for the aggregator machine. You may optionally include the IP address for the aggregator, replacing [IP_ADDRESS].

.. code-block:: console

  bash create-aggregator.sh AGGREGATOR.FULLY.QUALIFIED.DOMAIN.NAME

Tip: You can discover the FQDN with the Linux command:

.. code-block:: console

  hostname –fqdn

4.	For each test machine you want to run collaborators on, we create a collaborator certificate, replacing TEST.MACHINE.NAME with the actual test machine name. Note that this does not have to be the FQDN. Also, note that this script is run on the Aggregator node because it is the Aggregator that signs the certificate. Only Collaborators with valid certificates signed by the Aggregator can join the federation.

.. code-block:: console

  bash create-collaborator.sh TEST.MACHINE.NAME
