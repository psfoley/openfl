.. # Copyright (C) 2020 Intel Corporation
.. # Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

.. _running_the_federation_docker:

Running on Docker
#################
Please find below the steps to have the containerized fledge up&running and run the "docker_keras_demo.sh" demo.

## Build the docker image
======================
In order to successfully build the image, the Dockerfile is expecting to access the following dependencies:
- Find the "fledge" directory in the same location where we are going to execute the "docker build" command.
- Find the "docker_agg.sh" file
- Find the "docker_col.sh" file

### TODO: automate file retrieval --> need guidance 


```
$ docker build --build-arg http_proxy --build-arg https_proxy --build-arg socks_proxy --build-arg ftp_proxy --build-arg no_proxy -t fledge/docker -f fledge_containers/Dockerfile .
```

## Docker Demo
======================
### Short description
This demo runs on a single node and creates a federation with 2 institutions: 1 aggregator and 1 collaborator.
Both the institutions are containerized and the fledge sw-stack is self-contained within docker.

To  emulate the workspaces of both components, it will create 2 separated directories ("host_agg_workspace" and "host_col_workspace") in the /home/. on the local host.

The name of the docker image to be used for the demo can be set as first argument when calling the script. By default, the bash script will rely on the docker image name used to build it with the previous command: fledge/docker.

The path where the 2 local directories will be created can be passed as second argument. If empty, it will defautl to /home/$USERNAME.


### Run the demo
```
$ bash docker_keras_demo.sh
```

### Run the demo with custom params
```
$ bash docker_keras_demo.sh myDockerImg/name /My/Local/Path
```
