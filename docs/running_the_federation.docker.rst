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
  

Here's the :download:`"Hello Docker Federation <../docker/docker_keras_demo.sh>`. This is an end-to-end demo for the Keras CNN MNIST (:code:`docker_keras_demo.sh`).

.. literalinclude:: ../docker/docker_keras_demo.sh
  :language: bash
  


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
