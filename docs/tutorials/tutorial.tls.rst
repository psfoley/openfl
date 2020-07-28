.. # Copyright (C) 2020 Intel Corporation
.. # Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.


.. _tutorial-tls-pki:

(** THIS TUTORIAL NEEDS UPDATED FOLLOWING PKI NAME-CONVENTION CHANGES **)

Setting up PKI for TLS
----------------------

  The PKI consists of the following files on the following machines:

.. list-table:: PKI Files
   :widths: 25 25 25 25
   :header-rows: 1

   * - Description
     - Filename
     - Created By
     - Copied to
   * - Private key for CA, root key for federation
     - ca.key
     - Generating on Governor
     - None
   * - Public key/cert for CA
     - ca.crt
     - Generating on Governor
     - Every other node in federation
   * - CA serial file
     - ca.srl
     - Generating on Governor
     - None
   * - Private key for aggregator
     - local.key
     - Generating on aggregator
     - None
   * - Signing request for aggregator
     - local.csr
     - Generating on aggregator
     - Governor (for signing)
   * - Signed cert for aggregator
     - local.crt
     - Signing on Governor (requires local.csr)
     - Back to aggregator for use
   * - Private key for collaborator (each node)
     - local.key
     - Generating on collaborator
     - None
   * - Signing request for collaborator
     - local.csr
     - Generating on collaborator
     - Governor (for signing)
   * - Signed cert for collaborator
     - local.crt
     - Signing on Governor (requires local.csr)
     - Back to specific collaborator for use
     
Therefore, we will need to create the ca and copy its public .crt file to each other node. On those other nodes, we will create a private key and a cert signing request (.csr file). The .csr files must be copied to the governor, signed and the output .crt files copied back to the corresponding aggregator/collaborator. NOTE: at the moment, aggregator/collaborator .csr files all have the same name, which will be error prone. We need to fix this.

NOTE: our openssl commands are consistent with openssl 1.0.2 command line interface, including all implied defaults. Older/newer versions with different interfaces/defaults could alter the behavior of these commands!

To create the CA, we run:

.. code-block:: console

    openssl genrsa -out bin/federations/certs/<fed_name>/ca.key 3072
    openssl req -new -x509 -key bin/federations/certs/test/ca.key -out bin/federations/certs/<fed_name>/ca.crt -subj "/CN=Trusted Federated Learning <fed_name> Cert Authority"

Now we should have ca.key and ca.crt. Copy the ca.crt file into each aggregator/collaborator under:

.. code-block:: console

    bin/federations/certs/<fed_name>/ca.crt
    
For each aggregator/collaborator node, we need to create the local key and csr:

.. code-block:: console

    openssl genrsa -out bin/federations/certs/<fed_name>/local.key 3072
    openssl req -new -key bin/federations/certs/<fed_name>/local.key -out bin/federations/certs/<fed_name>/local.csr -subj /CN=<full_hostname>
	
Copy the .csr file to the governor to any path (NOTE: If running the aggregator and governor on the same node, do not accidentally overwrite the aggregator cert files). Then, sign the .csr file with the following command:

.. code-block:: console

    openssl x509 -req -in <path_to>/local.csr -CA bin/federations/certs/<fed_name>/ca.crt -CAkey bin/federations/certs/<fed_name>/ca.key -CAcreateserial -out <path_to>/local.crt

Then copy the output .crt file back to the aggregator/collaborator under:

.. code-block:: console

    bin/federations/certs/<fed_name>/local.crt
