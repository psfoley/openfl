#!/bin/bash


# Parameter definition
MODE=${1:-'init'}                  # ['init',...,'start']

WORKSPACE_DIR=${2:-'fed_work12345alpha81671'}  # This can be whatever unique directory name you want
COL=${3:-'one123dragons'}                # This can be any unique label
FED_PATH=${4:-'/home/fledge'}      # FED_WORKSPACE Path
TEMPLATE=${5:-'keras_cnn_mnist'}   # ['torch_cnn_mnist', 'keras_cnn_mnist']


[[ ! -z "$FED_PATH" ]] && cd $FED_PATH



# Functions definition
init() {

    FED_WORKSPACE=$1
    FED_PATH=$2
    TEMPLATE=$3

    #ARCHIVE_NAME="${FED_WORKSPACE}.zip"
    CURRENT_DIR=`pwd`

    # Create FL workspace
    [[ ! -z "$FED_PATH" ]] && cd $FED_PATH
    fx workspace create --prefix $FED_WORKSPACE --template $TEMPLATE
    cd $WORKSPACE_DIR

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