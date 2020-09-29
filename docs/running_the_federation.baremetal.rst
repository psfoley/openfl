.. # Copyright (C) 2020 Intel Corporation
.. # Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

.. _running_baremetal:

Creating the Federation
#######################

TL;DR
~~~~~

Here's the "Hello Federation" :code:`bash` script used for testing the project pipeline.

.. code-block:: bash

    #!/bin/bash

    # Test the pipeline

    TEMPLATE=${1:-'keras_cnn_mnist'}  # ['torch_cnn_mnist', 'keras_cnn_mnist']
    FED_WORKSPACE=${2:-'fed_work12345alpha81671'}   # This can be whatever unique directory name you want
    COL1=${3:-'one123dragons'}  # This can be any unique label (lowercase)
    COL2=${4:-'beta34unicorns'} # This can be any unique label (lowercase)

    FQDN=$(hostname --all-fqdns | awk '{print $1}')

    create_collaborator() {

        FED_WORKSPACE=$1
        FED_DIRECTORY=$2
        COL=$3
        COL_DIRECTORY=$4

        ARCHIVE_NAME="${FED_WORKSPACE}.zip"

        # Copy workspace to collaborator directories (these can be on different machines)
        rm -rf ${COL_DIRECTORY}    # Remove any existing directory
        mkdir -p ${COL_DIRECTORY}  # Create a new directory for the collaborator
        cd ${COL_DIRECTORY}
        fx workspace import --archive ${FED_DIRECTORY}/${ARCHIVE_NAME} # Import the workspace to this collaborator

        # Create collaborator certificate 
        cd ${COL_DIRECTORY}/${FED_WORKSPACE}
        fx collaborator create -n ${COL} --silent # Remove '--silent' if you run this manually

        # Sign collaborator certificate 
        cd ${FED_DIRECTORY}  # Move back to the Aggregator
        fx collaborator certify --certificate_name ${COL_DIRECTORY}/${FED_WORKSPACE}/cert/col_${COL}/col_${COL}.csr --silent # Remove '--silent' if you run this manually

    }

    # START
    # =====
    # Make sure you are in a Python virtual environment with the FL package installed.

    # Create FL workspace
    rm -rf ${FED_WORKSPACE}
    fx workspace create --prefix ${FED_WORKSPACE} --template ${TEMPLATE}
    cd ${FED_WORKSPACE}
    FED_DIRECTORY=`pwd`  # Get the absolute directory path for the workspace

    # Initialize FL plan
    fx plan initialize -a ${FQDN}

    # Create certificate authority for workspace
    fx workspace certify

    # Export FL workspace
    fx workspace export

    # Create aggregator certificate
    fx aggregator create --fqdn ${FQDN}

    # Sign aggregator certificate
    fx aggregator certify --certificate_name cert/agg_${FQDN}/agg_${FQDN}.csr --silent # Remove '--silent' if you run this manually

    # Create collaborator #1
    COL1_DIRECTORY=${FED_DIRECTORY}/${COL1}
    create_collaborator ${FED_WORKSPACE} ${FED_DIRECTORY} ${COL1} ${COL1_DIRECTORY}

    # Create collaborator #2
    COL2_DIRECTORY=${FED_DIRECTORY}/${COL2}
    create_collaborator ${FED_WORKSPACE} ${FED_DIRECTORY} ${COL2} ${COL2_DIRECTORY}

    # # Run the federation
    if cd ${FED_DIRECTORY} & fx aggregator start & cd ${COL1_DIRECTORY} & fx collaborator start -n ${COL1} & cd ${COL2_DIRECTORY} & fx collaborator start -n ${COL2} ; then
       rm -rf ${FED_DIRECTORY}
    fi



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
