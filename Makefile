whl = dist/tfedlrn-0.0.0-py3-none-any.whl
tfl = venv/lib/python3.5/site-packages/tfedlrn

col_id ?= col_0

full_hostname = $(shell hostname).$(shell hostname -d)

.PHONY: ca
ca: bin/federations/certs/test/ca.crt bin/federations/certs/test/ca.key

.PHONY: local_certs
local_certs: bin/federations bin/federations/certs/test bin/federations/certs/test/local.csr bin/federations/certs/test/local.crt

.PHONY: wheel
wheel: $(whl)

.PHONY: install
install: $(tfl)

.PHONY: venv
venv: venv/bin/python3

venv/bin/python3:
	python3 -m venv venv
	venv/bin/pip3 install setuptools
	venv/bin/pip3 install wheel
	
$(whl): venv/bin/python3
	venv/bin/python3 setup.py bdist_wheel

$(tfl): $(whl)
	venv/bin/pip3 install $(whl)

bin/federations/certs/test:
	mkdir -p bin/federations/certs/test

bin/federations/weights/mnist_cnn_keras_init.pbuf:
	echo "recipe needed!"

# start_mnist_agg_no_tls: $(tfl) federations/weights/mnist_cnn_keras_init.pbuf
# 	venv/bin/python3 bin/grpc_aggregator.py --plan_path federations/plans/mnist_a.yaml --disable_tls --disable_client_auth

# start_mnist_col_no_tls: $(tfl) federations/weights/mnist_cnn_keras_init.pbuf
# 	venv/bin/python3 bin/grpc_collaborator.py --plan_path federations/plans/mnist_a.yaml --col_id $(col_id) --disable_tls --disable_client_auth

start_mnist_agg: $(tfl) bin/federations/weights/mnist_cnn_keras_init.pbuf local_certs
	cd bin && ../venv/bin/python3 run_aggregator_from_flplan.py -p mnist_a.yaml

start_mnist_col: $(tfl) bin/federations/weights/mnist_cnn_keras_init.pbuf local_certs
	cd bin && ../venv/bin/python3 run_collaborator_from_flplan.py -p mnist_a.yaml -col $(col_id)

bin/federations/certs/test/ca.key:
	openssl genrsa -out bin/federations/certs/test/ca.key 3072

bin/federations/certs/test/ca.crt: bin/federations/certs/test/ca.key
	openssl req -new -x509 -key bin/federations/certs/test/ca.key -out bin/federations/certs/test/ca.crt -subj "/CN=Trusted Federated Learning Test Cert Authority"

bin/federations/certs/test/local.key:
	openssl genrsa -out bin/federations/certs/test/local.key 3072

bin/federations/certs/test/local.csr: bin/federations/certs/test/local.key
	openssl req -new -key bin/federations/certs/test/local.key -out bin/federations/certs/test/local.csr -subj /CN=$(full_hostname)

bin/federations/certs/test/local.crt: bin/federations/certs/test/local.csr bin/federations/certs/test/ca.crt
	openssl x509 -req -in bin/federations/certs/test/local.csr -CA bin/federations/certs/test/ca.crt -CAkey bin/federations/certs/test/ca.key -CAcreateserial -out bin/federations/certs/test/local.crt

clean:
	rm -rf venv
	rm -rf dist
	rm -rf build
	rm -rf tfedlrn.egg-info
	rm -rf bin/federations/certs/test/*
