set -e

function KillJobs(){
    echo "jobs to kill::"
    echo "==================="
    jobs
    echo "==================="
    for job in $(jobs -p); do
        kill -s SIGTERM $job || (sleep 3 && kill -9 $job)
    done
}

function cleanup(){
    echo "cleaning the processes ...... exit code::$?"
    KillJobs
    echo "after cleaning the processes ...... exit code::$?"
    exit $?
}

trap cleanup SIGINT SIGTERM SIGQUIT

#echo "======== pki creating certificates ......"                                                              
#time ../venv/bin/python3 ../bin/create_pki_for_flplan.py -p keras_cnn_mnist_2_mpmm.yaml                            
#echo "-------------------------------------------------------------"  

echo "sleeping for 30 sec. waiting for the aggregator"
sleep 30

echo "-------------------------------------------------------------"
echo "======== collaborator 0 ......"
echo "-------------------------------------------------------------"
time ../venv/bin/python3 ../bin/run_collaborator_from_flplan.py -p keras_cnn_mnist_2_mpmm.yaml -col col_0 -scn spr-gpu02.jf.intel.com &
echo "======== exit code $?"

#echo "sleeping for 30 sec. waiting for the aggregator"
#sleep 30

echo "-------------------------------------------------------------"
echo "======== collaborator 1 ......"
echo "-------------------------------------------------------------"
time ../venv/bin/python3 ../bin/run_collaborator_from_flplan.py -p keras_cnn_mnist_2_mpmm.yaml -col col_1 -scn spr-gpu02.jf.intel.com &
wait
echo "======== Exit code $?"
echo "======== Done! on gpu02 ${0}"
