echo "running ... single process single machine"
echo "testing pytorch creating initial weights ..."
CUDA_VISIBLE_DEVICES=9 ../venv/bin/python3 ../bin/create_initial_weights_file_from_flplan.py -p pt_resnet_withcifar10.yaml
echo "testing pytorch training ..."
CUDA_VISILBE_DEVICES=9 ../venv/bin/python3 ../bin/run_simulation_from_flplan.py -p pt_resnet_withcifar10.yaml

echo "testing pytorch creating initial weights ..."
CUDA_VISIBLE_DEVICES=9 ../venv/bin/python3 ../bin/create_initial_weights_file_from_flplan.py -p tf_resnet_withcifar10.yaml
echo "testing pytorch training ..."
CUDA_VISILBE_DEVICES=9 ../venv/bin/python3 ../bin/run_simulation_from_flplan.py -p tf_cnn_withcifar10.yaml
