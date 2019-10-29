whl = dist/tfedlrn-0.0.0-py3-none-any.whl
tfl = venv/lib/python3.5/site-packages/tfedlrn

col_id ?= col_0

full_hostname = $(shell hostname).$(shell hostname -d)

.PHONY: ca
ca: federations/certs/test/ca.crt federations/certs/test/ca.key

.PHONY: local_certs
local_certs: federations/certs/test/local.csr federations/certs/test/local.crt

.PHONY: wheel
wheel: $(whl)

.PHONY: install
install: $(tfl)

.PHONY: venv
venv: venv/bin/python3

.PHONY: init_unet
init_unet: initial_models/TensorFlow2DUNet.pbuf

venv/bin/python3:
	python3.5 -m venv venv
	venv/bin/pip3 install setuptools
	venv/bin/pip3 install wheel
	
$(whl): venv/bin/python3
	venv/bin/python3 setup.py bdist_wheel

$(tfl): $(whl)
	venv/bin/pip3 install $(whl)

initial_models:
	mkdir initial_models

initial_models/TensorFlow2DUNet.pbuf: initial_models $(tfl)
	venv/bin/python3 bin/build_initial_tensorflow_model.py -m TensorFlow2DUNet
	
run_brats_unet_fed: initial_models/TensorFlow2DUNet.pbuf
	venv/bin/python3 bin/simple_fl_tensorflow_test.py -n 0 -m TensorFlow2DUNet

federations/weights/mnist_cnn_keras_init.pbuf:
	echo "recipe needed!"

# start_mnist_agg_no_tls: $(tfl) federations/weights/mnist_cnn_keras_init.pbuf
# 	venv/bin/python3 bin/grpc_aggregator.py --plan_path federations/plans/mnist_a.yaml --disable_tls --disable_client_auth

# start_mnist_col_no_tls: $(tfl) federations/weights/mnist_cnn_keras_init.pbuf
# 	venv/bin/python3 bin/grpc_collaborator.py --plan_path federations/plans/mnist_a.yaml --col_id $(col_id) --disable_tls --disable_client_auth

start_mnist_agg: $(tfl) federations/weights/mnist_cnn_keras_init.pbuf local_certs
	cd bin && ../venv/bin/python3 run_aggregator_from_flplan.py -p mnist_a.yaml

start_mnist_col: $(tfl) federations/weights/mnist_cnn_keras_init.pbuf local_certs
	cd bin && ../venv/bin/python3 run_collaborator_from_flplan.py -p mnist_a.yaml -col $(col_id)

federations/certs/test/ca.key:
	openssl genrsa -out federations/certs/test/ca.key 3072

federations/certs/test/ca.crt: federations/certs/test/ca.key
	openssl req -new -x509 -key federations/certs/test/ca.key -out federations/certs/test/ca.crt -subj "/CN=Trusted Federated Learning Test Cert Authority"

federations/certs/test/local.key:
	openssl genrsa -out federations/certs/test/local.key 3072

federations/certs/test/local.csr: federations/certs/test/local.key
	openssl req -new -key federations/certs/test/local.key -out federations/certs/test/local.csr -subj /CN=$(full_hostname)

federations/certs/test/local.crt: federations/certs/test/local.csr federations/certs/test/ca.crt
	openssl x509 -req -in federations/certs/test/local.csr -CA federations/certs/test/ca.crt -CAkey federations/certs/test/ca.key -CAcreateserial -out federations/certs/test/local.crt

clean:
	rm -rf venv
	rm -rf dist
	rm -rf build
	rm -rf tfedlrn.egg-info
