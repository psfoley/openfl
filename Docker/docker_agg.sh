#!/bin/bash


## PARAMETER SETTING
# System params
HOST_USER=${1:-''}
HOST_UID=${2:-''}
HOST_GID=${3:-''}


# Fledge params
MODE=${4:-'init'}                  # ['init',...,'start']

WORKSPACE_DIR=${5:-'fed_work12345alpha81671'}  # This can be whatever unique directory name you want
COL=${6:-'pippo12'}                # This can be any unique label
FED_PATH=${7:-'/home/fledge'}      # FED_WORKSPACE Path
TEMPLATE=${8:-'keras_cnn_mnist'}   # ['torch_cnn_mnist', 'keras_cnn_mnist']


if [ ! -z "$FED_PATH" ]; then
   groupadd -g ${HOST_GID} ${HOST_USER}
   useradd --no-log-init -m -u ${HOST_UID} -g ${HOST_GID} -s /bin/bash ${HOST_USER}
   chown -R ${HOST_USER}:${HOST_USER} ${FED_PATH}
   cd ${FED_PATH}

elif [ -d "$WORKSPACE_DIR" ] ; then
   groupadd -g ${HOST_GID} ${HOST_USER}
   useradd --no-log-init -m -u ${HOST_UID} -g ${HOST_GID} -s /bin/bash ${HOST_USER}
   chown -R ${HOST_USER}:${HOST_USER} ${FED_PATH}
else
   echo "Missing dir: no permissions to be set"

fi


## AUX METHODS
init() {

    FED_WORKSPACE=$1
    FED_PATH=$2
    TEMPLATE=$3

    #ARCHIVE_NAME="${FED_WORKSPACE}.zip"
    CURRENT_DIR=`pwd`

    # Create FL workspace
    [[ ! -z "$FED_PATH" ]] && cd ${FED_PATH}
    fx workspace create --prefix ${FED_WORKSPACE} --template ${TEMPLATE}
    cd ${WORKSPACE_DIR}

    # Initialize FL plan
    FQDN=$(hostname --all-fqdns | awk '{print $1}')
    fx plan initialize -a ${FQDN}

    # Create certificate authority for workspace
    fx workspace certify

    # Create aggregator certificate
    fx aggregator generate-cert-request --fqdn ${FQDN}

    # Sign aggregator certificate
    fx aggregator certify --fqdn ${FQDN} --silent # Remove '--silent' if you run this manually

    # Move back to initial dir
    cd $CURRENT_DIR
}


add_col() {

    FED_WORKSPACE=$1
    FED_PATH=$2
    COL=$3

    CURRENT_DIR=`pwd`
    [[ ! -z "$FED_PATH" ]] && cd $FED_PATH
    cd $FED_WORKSPACE
   
    fx collaborator certify --request-pkg cert/col_${COL}_to_agg_cert_request.zip --silent # Remove '--silent' if you run this manually

    # Move back to initial dir
    cd $CURRENT_DIR
}


export_ws() {

    FED_WORKSPACE=$1
    FED_PATH=$2

    CURRENT_DIR=`pwd`
    [[ ! -z "$FED_PATH" ]] && cd $FED_PATH
    cd $FED_WORKSPACE

    fx workspace export

    cd $CURRENT_DIR
}


start() {

    FED_WORKSPACE=$1
    FED_PATH=$2

    CURRENT_DIR=`pwd`

    [[ ! -z "$FED_PATH" ]] && cd $FED_PATH
    cd $FED_WORKSPACE
    C=`pwd`

    fx aggregator start

    cd $CURRENT_DIR
}



if [ "$MODE" == "init" ]; then

    init $WORKSPACE_DIR $FED_PATH $TEMPLATE

elif [ "$MODE" == "col" ]; then

    add_col $WORKSPACE_DIR $FED_PATH $COL

elif [ "$MODE" == "export" ]; then

    export_ws $WORKSPACE_DIR $FED_PATH

elif [ "$MODE" == "start" ]; then

    start $WORKSPACE_DIR $FED_PATH

else

    echo "Unrecognized Mode. Aborting"

fi


# Setting user permissions
#echo "STO PER SETTARLO PER IL CMD $MODE"
#if [ ! -z "$HOST_USER" ]; then

#    [ -d "$WORKSPACE_DIR" ] && chown -R ${HOST_USER}:${HOST_USER} ${WORKSPACE_DIR} && echo "SETTATO PER IL CMD $MODE"

#fi
#echo "AGG HO FINITO"
