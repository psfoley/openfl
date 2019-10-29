[![pipeline status](https://gitlab.devtools.intel.com/weilinxu/spr_secure_intelligence-trusted_federated_learning/badges/master/pipeline.svg)](https://gitlab.devtools.intel.com/weilinxu/spr_secure_intelligence-trusted_federated_learning/commits/master)
[![coverage report](https://gitlab.devtools.intel.com/weilinxu/spr_secure_intelligence-trusted_federated_learning/badges/master/coverage.svg)](https://gitlab.devtools.intel.com/weilinxu/spr_secure_intelligence-trusted_federated_learning/commits/master)



gRPC refactor:

To run an MNIST example with a single collaborator:

1. Edit federations/plans/mnist_a.yaml and set the aggregator/addr field to the IP of your aggregator.
2. Open a terminal on the aggregator, go to the root tfedlrn directory and run: make start_mnist_agg
3. Open a terminal on the collaborator, go to the root tfedlrn directory and run: make start_mnist_col
