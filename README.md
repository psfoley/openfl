[![pipeline status](https://gitlab.devtools.intel.com/weilinxu/spr_secure_intelligence-trusted_federated_learning/badges/master/pipeline.svg)](https://gitlab.devtools.intel.com/weilinxu/spr_secure_intelligence-trusted_federated_learning/commits/master)
[![coverage report](https://gitlab.devtools.intel.com/weilinxu/spr_secure_intelligence-trusted_federated_learning/badges/master/coverage.svg)](https://gitlab.devtools.intel.com/weilinxu/spr_secure_intelligence-trusted_federated_learning/commits/master)



gRPC refactor:

To run an MNIST example with a single collaborator:

1. Edit federations/plans/mnist_a.yaml and set the aggregator/addr field to the IP of your aggregator.
2. If not using TLS, set the tls/disable value to True.
3. Open a terminal on the aggregator, go to the root tfedlrn directory and run: make start_mnist_agg
4. Open a terminal on the collaborator, go to the root tfedlrn directory and run: make start_mnist_col
5. NOTE: if using TLS, the ca.crt must match between the machines. By default, we've included the "test" ca.crt in the repo and set the .yaml files to use this cert.
