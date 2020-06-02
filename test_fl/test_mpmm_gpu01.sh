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

echo "-------------------------------------------------------------"
echo "======== on gpu01 ......"
echo "======== creating init weights ......"
echo "-------------------------------------------------------------"
time ../venv/bin/python3 ../bin/create_initial_weights_file_from_flplan.py -p keras_cnn_mnist_2_mpmm.yaml

echo "-------------------------------------------------------------"
echo "======== aggregator on gpu01 ......"
echo "-------------------------------------------------------------"
time ../venv/bin/python3 ../bin/run_aggregator_from_flplan.py -p keras_cnn_mnist_2_mpmm.yaml &
wait
echo "-------------------------------------------------------------"
echo "======== Exit code:: $?"
echo "======== Done! on gpu01 ${0}"

