set -e
echo "running ... single process single machine"

echo "-------------------------------------------------------------"
echo "1======== cifar10, pytorch cnn creating initial weights ..."
echo "-------------------------------------------------------------"
time CUDA_VISIBLE_DEVICES=9 ../venv/bin/python3 ../bin/create_initial_weights_file_from_flplan.py -p pt_cnn_withcifar10.yaml
echo "======= cifar10, pytorch cnn training ..."
time CUDA_VISILBE_DEVICES=9 ../venv/bin/python3 ../bin/run_simulation_from_flplan.py -p pt_cnn_withcifar10.yaml

echo "-------------------------------------------------------------"
echo "2======= cifar10, keras cnn creating initial weights ..."
echo "-------------------------------------------------------------"
time CUDA_VISIBLE_DEVICES=9 ../venv/bin/python3 ../bin/create_initial_weights_file_from_flplan.py -p keras_cnn_withcifar10.yaml
echo "======== cifar10, keras cnn training ..."
time CUDA_VISILBE_DEVICES=9 ../venv/bin/python3 ../bin/run_simulation_from_flplan.py -p keras_cnn_withcifar10.yaml

echo "-------------------------------------------------------------"
echo "3======== cifar10, pytorch resnet creating initial weights ..."
echo "-------------------------------------------------------------"
time CUDA_VISIBLE_DEVICES=9 ../venv/bin/python3 ../bin/create_initial_weights_file_from_flplan.py -p pt_resnet_withcifar10.yaml
echo "======== cifar10, pytorch resent training ..."
time CUDA_VISILBE_DEVICES=9 ../venv/bin/python3 ../bin/run_simulation_from_flplan.py -p pt_resnet_withcifar10.yaml

echo "-------------------------------------------------------------"
echo "4======= cifar10, keras resnet creating initial weights ..."
echo "-------------------------------------------------------------"
time CUDA_VISIBLE_DEVICES=9 ../venv/bin/python3 ../bin/create_initial_weights_file_from_flplan.py -p tf_resnet_withcifar10.yaml
echo "======= cifar10, keras resnet training ..."
time CUDA_VISILBE_DEVICES=9 ../venv/bin/python3 ../bin/run_simulation_from_flplan.py -p tf_resnet_withcifar10.yaml

echo "======== Done! ${0}"
