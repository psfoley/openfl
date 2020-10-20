.. # Copyright (C) 2020 Intel Corporation
.. # Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

.. _install_docker:

Docker Installation
###################

.. note::
    You'll need Docker installed on all nodes. To check
    that Docker is installed and running properly, you
    can run the Docker *Hello World* command like this:

    .. code-block:: console

      $ docker run hello-world
      Hello from Docker!
      This message shows that your installation appears to be working correctly.
      ...
      ...
      ...

Build the docker image
======================

Requirements
~~~~~~~~~~~~

In order to successfully build the image, the Dockerfile is expecting to access the following dependencies:

* Find the :code:`fledge` directory in the same location where we are going to execute the :code:`docker build` command.
* Find the :code:`docker_agg.sh` file
* Find the :code:`docker_col.sh` file

Command
~~~~~~~

.. code-block:: console

   $ docker build --build-arg USERNAME=`whoami` --build-arg USER_ID=`id -u $HOST_USER` --build-arg GROUP_ID=`id -g $HOST_USER` -t fledge/docker -f fledge_containers/Dockerfile .