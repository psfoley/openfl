#!/bin/bash

# Parameter definition
MODE=${1:-'import'}                             # ['init','import_ws','import_crt','start']
WORKSPACE_DIR=${2:-'fed_work12345alpha81671'}   # This can be whatever unique$
COL=${3:-'one123dragons'}                       # This can be any unique label
FED_PATH=${4:-'/home/fledge'}                   # FED_WORKSPACE Path


[[ ! -z "$FED_PATH" ]] && FED_WORKSPACE=${FED_PATH}/${WORKSPACE_DIR} || FED_WORKSPACE=$WORKSPACE_DIR


# Methods 
init() {

    ## Agg will need to validate the outcome of this method ##

    FED_WORKSPACE=$1
    FED_PATH=$2
    COL=$3
    DATA_PATH=$4

    CURRENT_DIR=`pwd`
    [[ ! -z "$FED_PATH" ]] && cd $FED_PATH
    cd $FED_WORKSPACE

    ## CREATING COLLABORATORs
    fx collaborator generate-cert-request -d ${DATA_PATH} -n ${COL} --silent # Remove '--silent' if you run this manually

    # Move back to initial dir
    cd $CURRENT_DIR
}



import_ws() {

    FED_WORKSPACE=$1
    FED_PATH=$2

    ARCHIVE_NAME="${FED_WORKSPACE}.zip"
   
    CURRENT_DIR=`pwd`
    [[ ! -z "$FED_PATH" ]] && cd $FED_PATH

    fx workspace import --archive ${ARCHIVE_NAME} # Import the workspace to this collaborator

    # Move back to initial dir
    cd $CURRENT_DIR
}


import_crt() {

    FED_WORKSPACE=$1
    FED_PATH=$2

    ARCHIVE_NAME="agg_to_col_${COL}_signed_cert.zip"

    CURRENT_DIR=`pwd`
    [[ ! -z "$FED_PATH" ]] && cd $FED_PATH

    fx collaborator certify --import ${FED_DIRECTORY}/${ARCHIVE_NAME}

    # Move back to initial dir
    cd $CURRENT_DIR
}


start() {

    FED_WORKSPACE=$1
    FED_PATH=$2
    COL=$3

    CURRENT_DIR=`pwd`

    [[ ! -z "$FED_PATH" ]] && cd $FED_PATH
    cd $FED_WORKSPACE
    C=`pwd`

    fx collaborator start -n $COL

    # Move back to initial dir
    cd $CURRENT_DIR
}



if [ "$MODE" == "init" ]; then

    init $WORKSPACE_DIR $FED_PATH $COL 1

elif [ "$MODE" == "import_ws" ]; then

    import_ws $WORKSPACE_DIR $FED_PATH

elif [ "$MODE" == "import_crt" ]; then

    import_crt $WORKSPACE_DIR $FED_PATH


elif [ "$MODE" == "start" ]; then

    start $WORKSPACE_DIR $FED_PATH $COL

else

    echo "Unrecognized Mode. Aborting"

fi