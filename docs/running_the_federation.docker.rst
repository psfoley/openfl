.. # Copyright (C) 2020 Intel Corporation
.. # Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

.. _running_the_federation_docker:

Running on Docker
#################

First make sure you have :ref:`followed the Docker installation steps <docker_install>` to have the containerized version of |productName|. A demo script can be found at :code:`docker_keras_demo.sh`.

Docker Demo
===========

Short description
~~~~~~~~~~~~~~~~~

This demo runs on a single node and creates a federation with 2 institutions: 1 aggregator and 1 collaborator.
Both the institutions are containerized and the |productName| software stack is self-contained within docker.

To emulate the workspaces of both components, it will create 2 separated directories ("host_agg_workspace" and "host_col_workspace") in the /home/. on the local host.

The name of the docker image to be used for the demo can be set as first argument when calling the script. By default, the bash script will rely on the docker image name used to build it with the previous command (*e.g.* fledge/docker).

The path where the 2 local directories will be created can be passed as second argument. If empty, it will defautl to /home/$USERNAME.

.. code-block:: console
   $ bash docker_keras_demo.sh

Run the demo with custom parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can run the same Docker container and pass your custom image name and path names as follows:

.. code-block:: console

   $ bash docker_keras_demo.sh myDockerImg/name /My/Local/Path
