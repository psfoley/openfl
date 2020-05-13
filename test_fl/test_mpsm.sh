
#make -C ../ build_containers
#make -C ../ run_agg_container &
#make -C ../ run_col_container &


echo "======== pki ......"
../venv/bin/python3 ../bin/create_pki_for_flplan.py -p keras_cnn_mnist_2.yaml
echo "======== init weights ......"
../venv/bin/python3 ../bin/create_initial_weights_file_from_flplan.py -p keras_cnn_mnist_2.yaml

echo "======== agg ......"
../venv/bin/python3 ../bin/run_aggregator_from_flplan.py -p keras_cnn_mnist_2.yaml &
echo "======== col0 ......"
../venv/bin/python3 ../bin/run_collaborator_from_flplan.py -p keras_cnn_mnist_2.yaml -col col_0 &
echo "======== col1 ......"
../venv/bin/python3 ../bin/run_collaborator_from_flplan.py -p keras_cnn_mnist_2.yaml -col col_1
