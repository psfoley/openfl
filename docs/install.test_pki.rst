.. # Copyright (C) 2020 Intel Corporation
.. # Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

**************************
Creating the PKI for Development and Testing (WIP)
**************************

`TLS <https://en.wikipedia.org/wiki/Transport_Layer_Security>`_ encryption is
used for the network connections.
Therefore, security keys and certificates will need to be created for the
aggregator and collaborators
to negotiate the connection securely.

While it is possible to disable TLS for development, we recommend leaving it enabled for testing and IT reasons. For example, many networks will block outgoing unencrypted traffic. Testing without the network is better done via the simulator (TODO: link to simulator).

Production vs. Testing PKI
#########

There are two primary differences between a production PKI and the test PKI we will create here:

1. The root `certificate authority <https://en.wikipedia.org/wiki/Certificate_authority>`_ in this test PKI is created by you, rather than an existing certificate authority, so the trust is rooted in you.
2. To simplify changes and 