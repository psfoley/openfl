.. # Copyright (C) 2020 Intel Corporation
.. # Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

MNIST Examples
---------------
First, let's enter the example code folder:

.. code-block:: console

    cd bin/

We would need to run an instance of aggregator and several instances of collaborator to start a federated learning task.
The examples here only demonstrate the single-collaborator cases for simplicity.

Federated MNIST without TLS:

.. code-block:: console

    $ python grpc_aggregator.py --plan_path federations/plans/mnist_a.yaml --disable_tls
    $ python grpc_collaborator.py --plan_path federations/plans/mnist_a.yaml --col_id 0 --disable_tls

Federated MNIST without client-side TLS authentication:

.. code-block:: console

    python grpc_aggregator.py --plan_path federations/plans/mnist_a.yaml --disable_client_auth --ca=files/grpc/localhost.crt --certificate=files/grpc/spr-gpu02.jf.intel.com.crt --private_key=files/grpc/private/spr-gpu02.jf.intel.com.key
    python grpc_collaborator.py --plan_path federations/plans/mnist_a.yaml --col_id 0 --disable_client_auth --ca=files/grpc/localhost.crt

Federated MNIST with full TLS:

.. code-block:: console

    python grpc_aggregator.py --plan_path federations/plans/mnist_a.yaml --ca=files/grpc/localhost.crt --certificate=files/grpc/spr-gpu02.jf.intel.com.crt --private_key=files/grpc/private/spr-gpu02.jf.intel.com.key
    python grpc_collaborator.py --plan_path federations/plans/mnist_a.yaml --col_id 0 --ca=files/grpc/localhost.crt --certificate=files/grpc/10.24.14.200.crt --private_key=files/grpc/private/10.24.14.200.key

