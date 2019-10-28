Tutorials
*********

Run a simple MNIST example without TLS.
---------------
Enter the example code folder:
    cd bin/

Start the aggregator:
    python grpc_aggregator.py --plan_path federations/plans/mnist_a.yaml --disable_tls


Start the collaborator:
    python grpc_collaborator.py --plan_path federations/plans/mnist_a.yaml --col_id 0 --disable_tls