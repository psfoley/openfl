.. # Copyright (C) 2020 Intel Corporation
.. # Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

*****************
Design Philosophy
*****************

The overall design is that all of the scripts are built off of the
federation plan. The plan is just a `YAML <https://en.wikipedia.org/wiki/YAML>`_
file that defines the
collaborators, aggregator, connections, models, data,
and any other parameters that describes how the training will evolve.
In the “Hello Federation” demos, the plan will be located in the
YAML file: *bin/federations/plans/keras_cnn_mnist_2.yaml*.
As you modify the demo to meet your needs, you’ll effectively
just be modifying the plan along with the Python code defining
the model and the data loader in order to meet your requirements.
Otherwise, the same scripts will apply. When in doubt,
look at the FL plan’s YAML file.
