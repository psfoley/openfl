.. # Copyright (C) 2020 Intel Corporation
.. # Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

**************************
Configuring the Federation
**************************

`TLS <https://en.wikipedia.org/wiki/Transport_Layer_Security>`_ encryption is
used for the network connections.
Therefore, security keys and certificates will need to be created for the
aggregator and collaborators
to negotiate the connection securely. For the :ref:`Hello Federation <running_the_federation>` demo
we will run the aggregator and collaborators on the same localhost server
so these configuration steps just need to be done once on that machine.

.. _install_certs:

All Nodes
#########

1.	 Unzip the source code

.. code-block:: console

  $ unzip spr_secure_intelligence-trusted_federated_learning.zip

2.	Change into the project directory.

.. code-block:: console

  $ cd spr_secure_intelligence-trusted_federated_learning

On the Aggregator Node
######################

1.	Change the directory to bin/federations/pki:

.. code-block:: console

  $ cd bin/federations/pki

2.	Run the Certificate Authority script. This will setup the Aggregator node
as the `Certificate Authority <https://en.wikipedia.org/wiki/Certificate_authority>`_
for the Federation. All certificates will be
signed by the aggregator. Follow the command-line instructions and enter
in the information as prompted. The script will create a simple database
file to keep track of all issued certificates.

.. code-block:: console

  $ bash setup_ca.sh

3.	Run the aggregator cert script, replacing AGGREGATOR.FULLY.QUALIFIED.DOMAIN.NAME
with the actual `fully qualified domain name (FQDN) <https://en.wikipedia.org/wiki/Fully_qualified_domain_name>`_
for the aggregator machine. You may optionally include the
IP address for the aggregator, replacing [IP_ADDRESS].

.. code-block:: console

  $ bash create-aggregator.sh AGGREGATOR.FULLY.QUALIFIED.DOMAIN.NAME

.. note::
   You can discover the FQDN with the Linux command:

   .. code-block:: console

     $ hostname –-fqdn

4.	For each test machine you want to run collaborators on, we create a collaborator
certificate, replacing TEST.MACHINE.NAME with the actual test machine name.
Note that this does not have to be the FQDN. Also, note that this script
is run on the Aggregator node because it is the Aggregator that signs the
certificate. Only Collaborators with valid certificates signed by
the Aggregator can join the federation.

.. code-block:: console

  $ bash create-collaborator.sh TEST.MACHINE.NAME

5.	Once you have the certificates created, you need to move the certificates
to the correct machines and ensure each machine has the cert_chain.crt
needed to verify certificate signatures.
For example, on a test machine named TEST_MACHINE that
you want to be able to run as a collaborator, you should have:

+---------------------------+--------------------------------------------------------------+
| File Type                 | Filename                                                     |
+===========================+==============================================================+
| Certificate chain         | bin/federations/pki/cert_chain.crt                           |
+---------------------------+--------------------------------------------------------------+
| Collaborator certificate  | bin/federations/pki/col_TEST_MACHINE/col_TEST_MACHINE.crt    |
+---------------------------+--------------------------------------------------------------+
| Collaborator key          | bin/federations/pki/col_TEST_MACHINE/col_TEST_MACHINE.key    |
+---------------------------+--------------------------------------------------------------+

Note that once the certificates are transferred to the collaborator,
it is now possible
to participate in any future federations run by this aggregator.
(The aggregator can revoke this privilege.)

6.	On the aggregator machine you should have the files:

+---------------------------+------------------------------------------------------------------------------------------------------------------+
| File Type                 | Filename                                                                                                         |
+===========================+==================================================================================================================+
| Certificate chain         | bin/federations/pki/cert_chain.crt                                                                               |
+---------------------------+------------------------------------------------------------------------------------------------------------------+
| Aggregator certificate    | bin/federations/pki/agg_AGGREGATOR.FULLY.QUALIFIED.DOMAIN.NAME/agg_AGGREGATOR.FULLY.QUALIFIED.DOMAIN.NAME.crt    |
+---------------------------+------------------------------------------------------------------------------------------------------------------+
| Aggregator key            | bin/federations/pki/agg_AGGREGATOR.FULLY.QUALIFIED.DOMAIN.NAME/agg_AGGREGATOR.FULLY.QUALIFIED.DOMAIN.NAME.key    |
+---------------------------+------------------------------------------------------------------------------------------------------------------+
