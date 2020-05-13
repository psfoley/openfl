echo "======== pki ......"
../venv/bin/python3 ../bin/create_pki_for_flplan.py -p keras_cnn_mnist_2.yaml
echo "======== init weights ......"
../venv/bin/python3 ../bin/create_initial_weights_file_from_flplan.py -p keras_cnn_mnist_2.yaml

echo "-------------------------------------------------------------"
echo "======== agg ......"
echo "-------------------------------------------------------------"
../venv/bin/python3 ../bin/run_aggregator_from_flplan.py -p keras_cnn_mnist_2.yaml &
echo "-------------------------------------------------------------"
echo "======== col0 ......"
echo "-------------------------------------------------------------"
../venv/bin/python3 ../bin/run_collaborator_from_flplan.py -p keras_cnn_mnist_2.yaml -col col_0 &
echo "-------------------------------------------------------------"
echo "======== col1 ......"
echo "-------------------------------------------------------------"
time ../venv/bin/python3 ../bin/run_collaborator_from_flplan.py -p keras_cnn_mnist_2.yaml -col col_1
echo "======== Done! ${0}"
