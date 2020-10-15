#!/bin/bash

### VAR definition
## System var
HOST_USER=`whoami`
HOST_UID=`id -u $HOST_USER`
HOST_GID=`id -g $HOST_USER`


## Fledge var
WORKSPACE_DIR=${2:-'fed_work12345alpha81671'} 	# This can be whatever unique directory name you want 
#COL=${3:-'one123dragons'} 			# This can be any unique label 
COL=${3:-'pippo12'}
##FED_PATH=${4:-'/home/fledge'} 		# Federation workspace PATH within Docker 
TEMPLATE=${5:-'keras_cnn_mnist'}		# ['torch_cnn_mnist', 'keras_cnn_mnist']


## Local var
HOST_WORKSPACE=${1:-'/home/$HOST_USER'}
HOST_AGG=${HOST_WORKSPACE}/host_agg_workspace
HOST_COL=${HOST_WORKSPACE}/host_col_workspace

DOCKER_IMG=${2:-"fledge/docker"}flocker/user



## Prepare working env mkdir -p $HOST_WORKSPACE/host_agg_workspace 
mkdir -p ${HOST_AGG}
mkdir -p ${HOST_COL}


### AGGREGATOR
## Create workspace
docker run --rm -it --network=host -v ${HOST_AGG}/:/home/fledge ${DOCKER_IMG} /bin/bash -c "bash docker_agg.sh ${HOST_USER} ${HOST_UID} ${HOST_GID} init"
## Export workspace
docker run --rm -it --network=host -v ${HOST_AGG}/:/home/fledge ${DOCKER_IMG} /bin/bash -c "bash docker_agg.sh ${HOST_USER} ${HOST_UID} ${HOST_GID} export"


## Copy workspace from AGGREGATOR to COLLABORATOR directories
CMD=`cp ${HOST_AGG}/${WORKSPACE_DIR}/${WORKSPACE_DIR}.zip ${HOST_COL}/.`
if ${CMD}; then
     echo “Success”
else
     echo “Failure 1, exit status: $?”
     exit
fi


### COLLABORATOR
## Import workspace
docker run --rm -it --network=host -v ${HOST_COL}/:/home/fledge ${DOCKER_IMG} /bin/bash -c "bash docker_col.sh ${HOST_USER} ${HOST_UID} ${HOST_GID} import_ws"
## Initialize collaborato
docker run --rm -it --network=host -v ${HOST_COL}/:/home/fledge ${DOCKER_IMG} /bin/bash -c "bash docker_col.sh ${HOST_USER} ${HOST_UID} ${HOST_GID} init"


## Send COLLABORATOR request to AGGREGATOR 
CMD=`cp -r ${HOST_COL}/${WORKSPACE_DIR}/col_${COL}_to_agg_cert_request.zip ${HOST_AGG}/${WORKSPACE_DIR}/cert/.`
if ${CMD}; then
     echo “Success”
else
     echo “Failure2, exit status: $?”
     exit
fi




### AGGREGATOR
## Certify collaborator
docker run --rm -it --network=host -v ${HOST_AGG}:/home/fledge ${DOCKER_IMG} /bin/bash -c "bash docker_agg.sh ${HOST_USER} ${HOST_UID} ${HOST_GID} col"


## Send verified certificate from AGGREGATOR to COLLABORATOR
CMD=`cp ${HOST_AGG}/${WORKSPACE_DIR}/agg_to_col_${COL}_signed_cert.zip ${HOST_COL}/${WORKSPACE_DIR}/.`
if ${CMD}; then
     echo “Success”
else
     echo “Failure3, exit status: $?”
     exit
fi


### COLLABORATOR
## Import certificate
docker run --rm -it --network=host -v ${HOST_COL}:/home/fledge ${DOCKER_IMG} /bin/bash -c "bash docker_col.sh ${HOST_USER} ${HOST_UID} ${HOST_GID} import_crt"

### AGGREGATOR
## Start the aggregator
docker run --rm -it -d --network=host -v ${HOST_AGG}:/home/fledge ${DOCKER_IMG} /bin/bash -c "bash docker_agg.sh ${HOST_USER} ${HOST_UID} ${HOST_GID} start"

### COLLABORATOR
## Start the collaborator
docker run --rm -it --network=host -v ${HOST_COL}:/home/fledge ${DOCKER_IMG} /bin/bash -c "bash docker_col.sh ${HOST_USER} ${HOST_UID} ${HOST_GID} start"











