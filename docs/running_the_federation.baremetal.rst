.. # Copyright (C) 2020 Intel Corporation
.. # Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

.. _running_baremetal:

Creating the Federation
#######################

TL;DR
~~~~~

Here's the "Hello Federation" :code:`bash` :download:`script <../tests/gitlab/test_hello_federation.sh>` used for testing the project pipeline.

Hello Federation - Your First Federated Learning Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We will show you how to set up |productName| using a simple `MNIST <https://en.wikipedia.org/wiki/MNIST_database>`_
dataset and a `TensorFlow/Keras <https://www.tensorflow.org/>`_
CNN model as an example. In this "Hello Federation", there will be one aggregator and two collaborators. The same 
method described here can be modified to run any federation using |productName|.

Before you run the federation make sure you have installed |productName| 
:ref:`using these instructions <install_initial_steps>` on every node (i.e. aggregator and collaborators).

.. _creating_workspaces:

On the Aggregator
~~~~~~~~~~~~~~~~~

1. Make sure you have initialized the virtual environment and can run the :code:`fx` command.

2. Create a workspace for the new federation project.

    .. code-block:: console
    
       $ fx workspace create --prefix WORKSPACE.PATH --template keras_cnn_mnist
       
    where **WORKSPACE.PATH** is the directory to create your workspace. By specifying 
    the :code:`--template keras_cnn_mnist` the workspace will create a workspace 
    with a simple TensorFlow/Keras CNN model that will download the MNIST 
    dataset and train in a federation. A list of
    pre-created templates can be found by simply running the command:

    .. code-block:: console
    
       $ fx workspace create --prefix WORKSPACE.PATH 
       
    .. note::
    
    Existing TensorFlow models can either be copied into the :code:`code` subdirectory
    in the workspace or wrapped using the :code:`FLModel` class described in 
    the advanced tutorial.

3. Change to the workspace directory.

    .. code-block:: console
    
        $ cd WORKSPACE.PATH
     
        
4.  Although it is possible to train models from scratch, it is assumed that in many cases the federation may perform fine-tuning of a previously-trained model. For this reason, the pre-trained weights for the model will be stored within protobuf files on the aggregator and passed to the collaborators during initialization. As seen in the YAML file, the protobuf file with the initial weights is expected to be found in the file **keras_cnn_mnist_init.pbuf**. For this example, however, we’ll just create an initial set of random model weights and putting it into that file by running the command:

    .. code-block:: console
    
       $ fx plan initialize -a AFQDN

   where *AFQDN** is the actual `fully qualified domain name (FQDN) <https://en.wikipedia.org/wiki/Fully_qualified_domain_name>`_ of the aggregator node. If you omit this parameter, :code:`fx` will automatically use the FQDN of the current node assuming the node has been correctly set with a static address. 
   .. note::

   Each workspace may have multiple Federated Learning plans and multiple collaborator lists associated with it.
   Therefore, the Aggregator has the following optional parameters.

   +-------------------------+---------------------------------------------------------+
   | Optional Parameters     | Description                                             |
   +=========================+=========================================================+
   | -p, --plan_config PATH  | Federated Learning plan [default = plan/plan.yaml]      |
   +-------------------------+---------------------------------------------------------+
   | -c, --cols_config PATH  | Authorized collaborator list [default = plan/cols.yaml] |
   +-------------------------+---------------------------------------------------------+
   | -d, --data_config PATH  | The data set/shard configuration file                   |
   +-------------------------+---------------------------------------------------------+    
