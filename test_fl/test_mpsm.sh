set -e

function KillJobs(){
    echo "jobs to kill::"
    echo "==================="
    jobs -p
    echo "==================="
    for job in $(jos -p); do
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

time ../venv/bin/python3 ../bin/create_pki_for_flplan.py -p keras_cnn_mnist_2.yaml
echo "-------------------------------------------------------------"
echo "======== creating init weights ......"
echo "-------------------------------------------------------------"
time ../venv/bin/python3 ../bin/create_initial_weights_file_from_flplan.py -p keras_cnn_mnist_2.yaml

echo "-------------------------------------------------------------"
echo "======== aggregator ......"
echo "-------------------------------------------------------------"
time ../venv/bin/python3 ../bin/run_aggregator_from_flplan.py -p keras_cnn_mnist_2.yaml &
echo "-------------------------------------------------------------"
echo "======== collaborator 0 ......"
echo "-------------------------------------------------------------"
time ../venv/bin/python3 ../bin/run_collaborator_from_flplan.py -p keras_cnn_mnist_2.yaml -col col_0 &
echo "-------------------------------------------------------------"
echo "======== collaborator 1 ......"
echo "-------------------------------------------------------------"
time ../venv/bin/python3 ../bin/run_collaborator_from_flplan.py -p keras_cnn_mnist_2.yaml -col col_1 &
wait
echo "======== Done! ${0} ${?}"
