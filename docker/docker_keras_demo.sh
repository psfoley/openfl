#!/bin/bash

HOST_USER="$(whoami)"

### VAR definition
DOCKER_IMG=${1:-"fledge/docker"}
HOST_WORKSPACE=${2:-''}

if [  -z "$HOST_WORKSPACE" ]; then
   HOST_WORKSPACE=/home/${HOST_USER}
fi

## Fledge var
WORKSPACE_DIR=${3:-'fed_work12345alpha81671'} 	# This can be whatever unique directory name you want
COL=${4:-'one123dragons'} 			# This can be any unique label
FED_PATH=${5:-'/home/fledge'}    		# Federation workspace PATH within Docker
TEMPLATE=${6:-'keras_cnn_mnist'}		# ['torch_cnn_mnist', 'keras_cnn_mnist']

## Local var
HOST_AGG=${HOST_WORKSPACE}/host_agg_workspace
HOST_COL=${HOST_WORKSPACE}/host_col_workspace

## Prepare working env mkdir -p $HOST_WORKSPACE/host_agg_workspace
mkdir -p ${HOST_AGG}
mkdir -p ${HOST_COL}

### AGGREGATOR
## Create workspace
docker run --rm -it --network=host -v ${HOST_AGG}/:/home/fledge ${DOCKER_IMG} /bin/bash -c "bash docker_agg.sh init"
## Export workspace
docker run --rm -it --network=host -v ${HOST_AGG}/:/home/fledge ${DOCKER_IMG} /bin/bash -c "bash docker_agg.sh export"

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
docker run --rm -it --network=host -v ${HOST_COL}/:/home/fledge ${DOCKER_IMG} /bin/bash -c "bash docker_col.sh import_ws"
## Initialize collaborato
docker run --rm -it --network=host -v ${HOST_COL}/:/home/fledge ${DOCKER_IMG} /bin/bash -c "bash docker_col.sh init"

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
docker run --rm -it --network=host -v ${HOST_AGG}:/home/fledge ${DOCKER_IMG} /bin/bash -c "bash docker_agg.sh col"

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
docker run --rm -it --network=host -v ${HOST_COL}:/home/fledge ${DOCKER_IMG} /bin/bash -c "bash docker_col.sh import_crt"

### AGGREGATOR
## Start the aggregator
docker run --rm -it -d --network=host -v ${HOST_AGG}:/home/fledge ${DOCKER_IMG} /bin/bash -c "bash docker_agg.sh start"

### COLLABORATOR
## Start the collaborator
docker run --rm -it --network=host -v ${HOST_COL}:/home/fledge ${DOCKER_IMG} /bin/bash -c "bash docker_col.sh start"
