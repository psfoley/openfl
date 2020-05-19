set -e
#echo "======== pki creating certificates ......"
#time ../venv/bin/python3 ../bin/create_pki_for_flplan.py -p keras_cnn_mnist_2_mpmm.yaml                            

echo "-------------------------------------------------------------"
echo "======== on gpu01 ......"
echo "======== creating init weights ......"
echo "-------------------------------------------------------------"
time ../venv/bin/python3 ../bin/create_initial_weights_file_from_flplan.py -p keras_cnn_mnist_2_mpmm.yaml

echo "-------------------------------------------------------------"
echo "======== aggregator on gpu01 ......"
echo "-------------------------------------------------------------"
time ../venv/bin/python3 ../bin/run_aggregator_from_flplan.py -p keras_cnn_mnist_2_mpmm.yaml 
echo "-------------------------------------------------------------"
echo "======== Exit code:: $?"
echo "======== Done! on gpu01 ${0}"

