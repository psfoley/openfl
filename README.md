# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

[![pipeline status](https://gitlab.devtools.intel.com/weilinxu/spr_secure_intelligence-trusted_federated_learning/badges/master/pipeline.svg)](https://gitlab.devtools.intel.com/weilinxu/spr_secure_intelligence-trusted_federated_learning/commits/master)
[![coverage report](https://gitlab.devtools.intel.com/weilinxu/spr_secure_intelligence-trusted_federated_learning/badges/master/coverage.svg)](https://gitlab.devtools.intel.com/weilinxu/spr_secure_intelligence-trusted_federated_learning/commits/master)

Notes for learning how this codebase works:


*  docs/tutorial.docker.rst shows how to use docker containers to run a federation.
*  docs/tutorial.simulation.rst shows how to install into a virtualenv for running single-process simulations (the aggregator and collaborators run round-robin, and will even share the graph in serial fashion, correctly saving/restoring state).
*  To learn about setting up a federation and configuring it, start with the scripts in the /bin folder that are launched by the tutorials (run_aggregator_from_flplan.py and run_collaborator_from_flplan.py, or run_simulation_from_flplan.py). Then look at the .yaml files in /bin/federations/plans to see how federations are configured. 
*  To learn about porting a model to run in a federation, start with the flmodel.py and the related framework-specific sub-classes (pytorchflmodel.py, kerasflmodel.py, and tfflmodel.py)




