#trap "exit" INT TERM ERR
#trap "kill 0" EXIT
set -e

echo "running ... single process single machine"                                                                  
echo "stc-compression::creating initial weights ..."                                                                               
../venv/bin/python3 ../bin/create_initial_weights_file_from_flplan.py -p stc_compression_pipeline_test_mnist_10.yaml -c cols_10.yaml
echo "stc-compression::training ..."                                                                                               
../venv/bin/python3 ../bin/run_simulation_from_flplan.py -p stc_compression_pipeline_test_mnist_10.yaml -c cols_10.yaml

echo "skc-compression::creating initial weights ..."                                                                               
../venv/bin/python3 ../bin/create_initial_weights_file_from_flplan.py -p skc_compression_pipeline_test_mnist_10.yaml -c cols_10.yaml
echo "skc-compression::training ..."                                                                                               
../venv/bin/python3 ../bin/run_simulation_from_flplan.py -p skc_compression_pipeline_test_mnist_10.yaml -c cols_10.yaml

echo "kc-compression::creating initial weights ..."                                                                               
../venv/bin/python3 ../bin/create_initial_weights_file_from_flplan.py -p kc_compression_pipeline_test_mnist_10.yaml -c cols_10.yaml
echo "kc-compression::training ..."                                                                                               
../venv/bin/python3 ../bin/run_simulation_from_flplan.py -p kc_compression_pipeline_test_mnist_10.yaml -c cols_10.yaml
