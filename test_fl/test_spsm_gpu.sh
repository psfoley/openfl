set -e
echo "running ... single process single machine"

echo "-------------------------------------------------------------"
echo "1======== cifar10, pytorch cnn creating initial weights ..."
echo "-------------------------------------------------------------"
GPU=$(./find_free_gpu.sh)
echo "Using GPU $GPU"
CUDA_VISIBLE_DEVICES=$GPU time ../venv/bin/python3 ../bin/create_initial_weights_file_from_flplan.py -p pt_cnn_withcifar10.yaml

echo "======= cifar10, pytorch cnn training ..."
GPU=$(./find_free_gpu.sh)
echo "Using GPU $GPU"
CUDA_VISIBLE_DEVICES=$GPU time ../venv/bin/python3 ../bin/run_simulation_from_flplan.py -p pt_cnn_withcifar10.yaml

echo "-------------------------------------------------------------"
echo "2======= cifar10, keras cnn creating initial weights ..."
echo "-------------------------------------------------------------"
GPU=$(./find_free_gpu.sh)
echo "Using GPU $GPU"
CUDA_VISIBLE_DEVICES=$GPU time ../venv/bin/python3 ../bin/create_initial_weights_file_from_flplan.py -p keras_cnn_withcifar10.yaml

echo "======== cifar10, keras cnn training ..."
GPU=$(./find_free_gpu.sh)
echo "Using GPU $GPU"
CUDA_VISIBLE_DEVICES=$GPU time ../venv/bin/python3 ../bin/run_simulation_from_flplan.py -p keras_cnn_withcifar10.yaml

echo "-------------------------------------------------------------"
echo "3======== cifar10, pytorch resnet creating initial weights ..."
echo "-------------------------------------------------------------"
GPU=$(./find_free_gpu.sh)
echo "Using GPU $GPU"
CUDA_VISIBLE_DEVICES=$GPU time ../venv/bin/python3 ../bin/create_initial_weights_file_from_flplan.py -p pt_resnet_withcifar10.yaml

echo "======== cifar10, pytorch resent training ..."
GPU=$(./find_free_gpu.sh)
echo "Using GPU $GPU"
CUDA_VISIBLE_DEVICES=$GPU time ../venv/bin/python3 ../bin/run_simulation_from_flplan.py -p pt_resnet_withcifar10.yaml

echo "-------------------------------------------------------------"
echo "4======= cifar10, keras resnet creating initial weights ..."
echo "-------------------------------------------------------------"
GPU=$(./find_free_gpu.sh)
echo "Using GPU $GPU"
CUDA_VISIBLE_DEVICES=$GPU time ../venv/bin/python3 ../bin/create_initial_weights_file_from_flplan.py -p keras_resnet_withcifar10.yaml

echo "======= cifar10, keras resnet training ..."
GPU=$(./find_free_gpu.sh)
echo "Using GPU $GPU"
CUDA_VISIBLE_DEVICES=$GPU time ../venv/bin/python3 ../bin/run_simulation_from_flplan.py -p keras_resnet_withcifar10.yaml

echo "======== Done! ${0}"
