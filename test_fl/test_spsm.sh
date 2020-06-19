set -e
echo "running ... single process single machine"

echo "-------------------------------------------------------------"
echo "1======== mnist, pytorch cnn creating initial weights ..."
echo "-------------------------------------------------------------"
time ../venv/bin/python3 ../bin/create_initial_weights_file_from_flplan.py -p pt_cnn_withmnist.yaml
echo "======== mnist, pytorch cnn training ..."
time ../venv/bin/python3 ../bin/run_simulation_from_flplan.py -p pt_cnn_withmnist.yaml

echo "-------------------------------------------------------------"
echo "2======== mnist, keras cnn creating initial weights ..."
echo "-------------------------------------------------------------"
time ../venv/bin/python3 ../bin/create_initial_weights_file_from_flplan.py -p keras_cnn_withmnist.yaml
echo "======== mnist, keras cnn training ..."
time ../venv/bin/python3 ../bin/run_simulation_from_flplan.py -p keras_cnn_withmnist.yaml

echo "-------------------------------------------------------------"
echo "3======== mnist, pytorch resnet creating initial weights ..."
echo "-------------------------------------------------------------"
time ../venv/bin/python3 ../bin/create_initial_weights_file_from_flplan.py -p pt_resnet_withmnist.yaml
echo "======== mnist, pytorch resnet training ..."
time ../venv/bin/python3 ../bin/run_simulation_from_flplan.py -p pt_resnet_withmnist.yaml

echo "-------------------------------------------------------------"
echo "4======== mnist, keras resnet creating initial weights ..."
echo "-------------------------------------------------------------"
time ../venv/bin/python3 ../bin/create_initial_weights_file_from_flplan.py -p tf_resnet_withmnist.yaml
echo "======== mnist, keras resnet training ..."
time ../venv/bin/python3 ../bin/run_simulation_from_flplan.py -p tf_resnet_withmnist.yaml

echo "======== Done! ${0}"
