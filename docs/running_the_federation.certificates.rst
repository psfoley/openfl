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

    .. note::
    
    Certificates can be created for each project workspace.

.. _install_certs:

Before you run the federation make sure you have installed |productName| 
:ref:`using these instructions <install_initial_steps>` on every node (i.e. aggregator and collaborators), 
are in the correct Python virtual environment, and are in the correct directory for the :ref:`project workspace <creating_workspaces>`.


On the Aggregator Node
######################

1. Change directory to **WORKSPACE.PATH**:

    .. code-block:: console
    
       $ cd WORKSPACE.PATH

2. Run the Certificate Authority command. This will setup the Aggregator node as the `Certificate Authority <https://en.wikipedia.org/wiki/Certificate_authority>`_ for the Federation. All certificates will be signed by the aggregator. Follow the command-line instructions and enter in the information as prompted. The command will create a simple database file to keep track of all issued certificates. 

    .. code-block:: console
    
       $ fx workspace certify

3. Run the aggregator certify command, replacing **AGGREGATOR.FULLY.QUALIFIED.DOMAIN.NAME** with the actual `fully qualified domain name (FQDN) <https://en.wikipedia.org/wiki/Fully_qualified_domain_name>`_ for the aggregator machine. If you do not specify the IP address for the aggregator, then the current machine will be assumed to be the aggregator.

    .. code-block:: console
    
       $ fx aggregator certify --fqdn AGGREGATOR.FULLY.QUALIFIED.DOMAIN.NAME
       
    .. note::
    
    You can discover the FQDN with the Linux command:
    
        .. code-block:: console
        
           $ hostname --all-fqdns | awk '{print $1}'

4. For each test machine you want to run collaborators on, we create a collaborator certificate, replacing **TEST.MACHINE.NAME** with the actual test machine name. Note that this does not have to be the FQDN. Also, note that this command is run on the Aggregator node because it is the Aggregator that signs the certificate. Only Collaborators with valid certificates signed by the Aggregator can join the federation.

    .. code-block:: console
    
       $ fx collaborator certify -n TEST.MACHINE.NAME

5. Once you have the certificates created, you need to move the certificates to the correct machines and ensure each machine has the :code:`cert_chain.crt` needed to verify certificate signatures. For example, on a test machine named **TEST_MACHINE** that you want to be able to run as a collaborator, you should have:

    +---------------------------+--------------------------------------------------------------+
    | File Type                 | Filename                                                     |
    +===========================+==============================================================+
    | Certificate chain         | WORKSPACE.PATH/cert/cert_chain.crt                           |
    +---------------------------+--------------------------------------------------------------+
    | Collaborator certificate  | WORKSPACE.PATH/cert/col_TEST_MACHINE/col_TEST_MACHINE.crt    |
    +---------------------------+--------------------------------------------------------------+
    | Collaborator key          | WORKSPACE.PATH/cert/col_TEST_MACHINE/col_TEST_MACHINE.key    |
    +---------------------------+--------------------------------------------------------------+

6. On the aggregator machine you should have the files:

    +---------------------------+--------------------------------------------------+
    | File Type                 | Filename                                         |
    +===========================+==================================================+
    | Certificate chain         | WORKSPACE.PATH/cert/cert_chain.crt               |
    +---------------------------+--------------------------------------------------+
    | Aggregator certificate    | WORKSPACE.PATH/cert/agg_$AFQDN/agg_$AFQDN.crt    |
    +---------------------------+--------------------------------------------------+
    | Aggregator key            | WORKSPACE.PATH/cert/agg_$AFQDN/agg_$AFQDN.key    |
    +---------------------------+--------------------------------------------------+
    
    where **$AFQDN** is the fully-qualified domain name of the aggregator node.

