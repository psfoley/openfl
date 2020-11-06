.. # Copyright (C) 2020 Intel Corporation
.. # Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

.. _running_the_federation_docker:

Running on Docker
#################

First make sure you have :ref:`followed the Docker installation steps <install_docker>` to have the containerized version of |productName|. A demo script can be found at :code:`docker_keras_demo.sh`.


TL;DR
=====

Here's the :download:`DockerFile <../docker/Dockerfile>`. This image can be reused for aggregators and collaborators.

.. literalinclude:: ../docker/Dockerfile
  :language: docker
  

Here's the :download:`"Hello Docker Federation" <../docker/docker_keras_demo.sh>` demo. This is an end-to-end demo for the Keras CNN MNIST (:code:`docker_keras_demo.sh`).

.. literalinclude:: ../docker/docker_keras_demo.sh
  :language: bash

Custom execution
================

Once built, the current image can be instantiated in two modes:

As an **aggregator**:

.. code-block:: console

   $ container-home>> bash docker_agg.sh CMD
   
   
As a **collaborator**:

.. code-block:: console

   $ container-home>> bash docker_col.sh CMD
   
Each :code:`bash` file contains its own list of methods to implement and run the federation. Each method is relevant depending on which step of the pipeline one needs to address. 

+---------------------------------+--------------+-------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------+
| Command                         | Who          | What                                                                    | Assumptions                                                                                                   |
+=================================+==============+=========================================================================+===============================================================================================================+
| bash docker\_agg.sh init        | Aggregator   | Initialize and certify the workspace                                    | None                                                                                                          |
+---------------------------------+--------------+-------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------+
| bash docker\_agg.sh export      | Collaborator | Export the workspace into a "workspace\_name.zip" file                  | Workspace has been created                                                                                    |
+---------------------------------+--------------+-------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------+
| bash docker\_col.sh import\_ws  | Collaborator | Import the workspace "workspace\_name.zip"                              | File already transfered to the collaborator workspace directory on the host                                   |
+---------------------------------+--------------+-------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------+
| bash docker\_col.sh init        | Collaborator | initialize the collaborator (i.e. generates the "col\_$COL\_NAME" dir ) | Workspace has been exported                                                                                   |
+---------------------------------+--------------+-------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------+
| bash docker\_agg.sh col         | Aggregator   | certify the collaborator request                                        | "col\_$COL\_NAME" directory already transfered to the aggregator workspace directory                          |
+---------------------------------+--------------+-------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------+
| bash docker\_col.sh import\_crt | Collaborator | Import the validated certificate                                        | A signed certificate request zip archive has already been transfered to the aggregator                        |
+---------------------------------+--------------+-------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------+
| bash docker\_agg.sh start       | Aggregator   | Start the aggregator                                                    | None                                                                                                          |
+---------------------------------+--------------+-------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------+
| bash docker\_col.sh start       | Collaborator | Start the collabortor                                                   | The .crt files (collaborator and cert\_chain) have already been transfered to the correct collaborator        |
+---------------------------------+--------------+-------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------+

Execution on hosts with non-root access
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The current image ensures that both environment with root and non-root access can run the docker fledge container smoothly. 
While users with root access wouldn’t require particular instructions, there are few considerations that are worth to be shared for those user with limited permissions. 
To achieve this result, the image will need to be built by providing the information of the current user at build time. This will ensure that all the actions taken by the container at runtime, will be owned by the same user logged in the host.


Single and Multi-node execution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With the current image one can create a full federation within a single node or distributed across multiple nodes. 
The complete pipeline on how to initiate a federation composed by one aggregator and one collaborator running on the same node is demonstrated in :code:`docker_keras_demo.sh`.
The multinode execution has only been tested on an isolated cluster of three machines connected on the same internal network, respectively running one aggregator and two collaborators.
To simulate a realistic environment, these machines didn’t have password-less access between each other. The file exchanged between the aggregator and the collaborators at the beginning of the process (workspace, certificate requests and validated certificates) have been manually performed by coping the files from one host to the others. The mechanism to automate such operations is currently under consideration by the development team.
At this stage, one can replicate the approach adopted in the attached demo to run a custom federation. 


Hello Federation Docker
=======================

This demo runs on a single node and creates a federation with two institutions: one aggregator and one collaborator.
Both the institutions are containerized and the |productName| software stack is self-contained within docker.

To emulate the workspaces of both components, it will create two separate directories (*host_agg_workspace* and *host_col_workspace*) in the :code:`home` directory on the local host.

The name of the docker image to be used for the demo can be set as first argument when calling the script. By default, the bash script will rely on the docker image name used to build it with the previous command (*e.g.* :code:`fledge/docker`).

The path where the two local directories will be created can be passed as second argument. If empty, it will default to :code:`/home/$USERNAME`.

.. code-block:: console

   $ bash docker_keras_demo.sh

Run the demo with custom parameters
===================================

You can run the same Docker container and pass your custom image name and path names as follows:

.. code-block:: console

   $ bash docker_keras_demo.sh myDockerImg/name /My/Local/Path
