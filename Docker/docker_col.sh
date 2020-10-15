#!/bin/bash

## PARAMETER SETTING
# System params
HOST_USER=${1:-''}
HOST_UID=${2:-''}
HOST_GID=${3:-''}


# Fledge params
MODE=${4:-'import'}                # ['import', 'start']
WORKSPACE_DIR=${5:-'fed_work12345alpha81671'}  # This can be whatever unique$
COL=${6:-'pippo12'}                # This can be any unique label
FED_PATH=${7:-'/home/fledge'}      # FED_WORKSPACE Path


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

    fx collaborator certify --import ${FED_WORKSPACE}/${ARCHIVE_NAME}
    yes | cp -rf ${FED_PATH}/cert/* ${FED_WORKSPACE}/cert/.

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


#echo "COL STO PER SETTARLO PER IL CMD $MODE"
# Setting user permissions
#if [ ! -z "$HOST_USER" ]; then
#    groupadd -g ${HOST_GID} ${HOST_USER}
#    useradd --no-log-init -m -u ${HOST_UID} -g ${HOST_GID} -s /bin/bash ${HOST_USER}
#    [ -d "$WORKSPACE_DIR" ] && chown -R ${HOST_USER}:${HOST_USER} ${WORKSPACE_DIR} && echo "DENTROOOOOOO COL SETTATO PER IL CMD $MODE"
#fi
#echo "COL FINITO"
