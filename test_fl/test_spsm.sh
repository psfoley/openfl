
echo "running ... single process single machine"
echo "testing pytorch creating initial weights ..."
../venv/bin/python3 ../bin/create_initial_weights_file_from_flplan.py -p pt_cnn_withmnist.yaml
echo "testing pytorch training ..."
../venv/bin/python3 ../bin/run_simulation_from_flplan.py -p pt_cnn_withmnist.yaml
