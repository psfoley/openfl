
#echo "======== pki creating certificates ......"
#time ../venv/bin/python3 ../bin/create_pki_for_flplan.py -p keras_cnn_mnist_2_mpmm.yaml                            

echo "-------------------------------------------------------------"
echo "======== aggregator on gpu01 ......"
echo "-------------------------------------------------------------"
time ../venv/bin/python3 ../bin/run_aggregator_from_flplan.py -p keras_cnn_mnist_2_mpmm.yaml 
echo "-------------------------------------------------------------"
echo "======== Done! on gpu01 ${0}"

