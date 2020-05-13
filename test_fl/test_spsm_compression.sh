

echo "running ... single process single machine"                                                                  
echo "creating initial weights ..."                                                                               
../venv/bin/python3 ../bin/create_initial_weights_file_from_flplan.py -p stc_compression_pipeline_test_mnist_10.yaml
echo "training ..."                                                                                               
../venv/bin/python3 ../bin/run_simulation_from_flplan.py -p stc_compression_pipeline_test_mnist_10.yaml

