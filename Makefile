# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

# WIP for transfering tutorial steps to makefile

col_num ?= 0
use_gpu ?= false
dataset ?= mnist
python_version ?= python3.6
mount_type ?= rw

whl = dist/tfedlrn-0.0.0-py3-none-any.whl
tfl = venv/lib/$(python_version)/site-packages/tfedlrn

ifeq ($(use_gpu), true)
	base_image = tensorflow/tensorflow:1.14.0-gpu-py3
	device = gpu
	runtime_line = --runtime nvidia
else
	base_image = ubuntu:18.04
	device = cpu
endif

ifeq ($(dataset),brats)
    additional_brats_container_lines = \
	-v '/raid/datasets/BraTS17/symlinks/$(col_num)':/home/$(shell whoami)/tfl/datasets/brats:ro \
    -v '/raid/datasets/BraTS17/MICCAI_BraTS17_Data_Training/HGG':/raid/datasets/BraTS17/MICCAI_BraTS17_Data_Training/HGG:ro
endif



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
	$(python_version) -m venv venv
	# specifying pip version to avoid issue with later intel-tensorflow install
	venv/bin/pip3 install --force-reinstall pip==19.0
	venv/bin/pip3 install --upgrade setuptools
	venv/bin/pip3 install --upgrade wheel
	venv/bin/pip3 install --upgrade pyyaml

$(whl): venv/bin/python3
	venv/bin/python3 setup.py bdist_wheel
	# we will use the wheel, and do not want the egg info
	rm -r -f tfedlrn.egg-info

$(tfl): $(whl)
	venv/bin/pip3 install $(whl)

uninstall:
	venv/bin/pip3 uninstall -y tfedlrn
	rm -rf dist
	rm -rf build

.PHONY: reinstall
reinstall: uninstall install

bin/federations/certs/test:
	mkdir -p bin/federations/certs/test

bin/federations/weights/keras_cnn_mnist_init.pbuf:
	echo "recipe needed!"

# start_mnist_agg_no_tls: $(tfl) federations/weights/mnist_cnn_keras_init.pbuf
# 	venv/bin/python3 bin/grpc_aggregator.py --plan_path federations/plans/mnist_a.yaml --disable_tls --disable_client_auth

# start_mnist_col_no_tls: $(tfl) federations/weights/mnist_cnn_keras_init.pbuf
# 	venv/bin/python3 bin/grpc_collaborator.py --plan_path federations/plans/mnist_a.yaml --col_num $(col_num) --disable_tls --disable_client_auth

start_mnist_agg: $(tfl) bin/federations/weights/keras_cnn_mnist_init.pbuf local_certs
	cd bin && ../venv/bin/python3 run_aggregator_from_flplan.py -p keras_cnn_mnist_2.yaml

start_mnist_col: $(tfl) bin/federations/weights/keras_cnn_mnist_init.pbuf local_certs
	cd bin && ../venv/bin/python3 run_collaborator_from_flplan.py -p keras_cnn_mnist_2.yaml -col col_$(col_num)

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
	rm -r -f venv
	rm -r -f dist
	rm -r -f build
	rm -r -f bin/federations/certs/test/*


# ADDING TUTORIAL TARGETS
# simply throw an error if these recipies are called without a plan defined
ifndef plan
build_containers: venv/bin/python3
	$(error plan needs to be defined in order to run this recipe)
run_agg_container: venv/bin/python3
	$(error plan needs to be defined in order to run this recipe)
run_col_conatiner: venv/bin/python3
	$(error plan needs to be defined in order to run this recipe)
else
build_containers: venv/bin/python3
	# parse the flplan to obtain model info
	$(eval module_name=$(shell venv/bin/python3 bin/flplan_info_to_stdout.py \
	            -p $(plan) -fk model -sk module_name) )
	$(eval model_dir=$(shell echo $(module_name) | awk '{FS="." ; OFS="/"; $$0=$$0}  { print $$1,$$2,$$3}'))
	$(eval model=$(shell echo $(module_name) | awk '{FS="." ; $$0=$$0}  { print $$4}'))
	@echo building agg and coll containers for model whose module is named: $(model), and located in: $(model_dir)


	docker build \
	--build-arg BASE_IMAGE=$(base_image) \
	--build-arg http_proxy \
	--build-arg https_proxy \
	--build-arg socks_proxy \
	--build-arg ftp_proxy \
	--build-arg no_proxy \
	--build-arg UID=$(shell id -u) \
	--build-arg GID=$(shell id -g) \
	--build-arg UNAME=$(shell whoami) \
	-t tfl_agg_$(model)_$(shell whoami):0.1 \
	-f Dockerfile \
	.

	docker build --build-arg whoami=$(shell whoami) \
	--build-arg use_gpu=$(use_gpu) \
	-t tfl_col_$(device)_$(model)_$(shell whoami):0.1 \
	-f ./$(model_dir)/$(device).dockerfile \
	.

run_agg_container: venv/bin/python3
	# parse the flplan to obtain model info
	$(eval module_name=$(shell venv/bin/python3 bin/flplan_info_to_stdout.py \
	            -p $(plan) -fk model -sk module_name) )
	$(eval model=$(shell echo $(module_name) | awk '{FS="." ; $$0=$$0}  { print $$4}'))
	@echo running agg container for model contained in module named: $(model)

	@echo "Aggregator Container started."
	@docker run \
	--net=host \
	-it --name=tfl_agg_$(model)_$(shell whoami) \
	--rm \
	-w /home/$(shell whoami)/tfl/bin \
	-v $(shell pwd)/bin/federations:/home/$(shell whoami)/tfl/bin/federations:$(mount_type) \
	$(additional_brats_container_lines) \
	tfl_agg_$(model)_$(shell whoami):0.1 \
	bash -c "echo \"export PS1='\e[0;31m[FL Docker for \e[0;32mAggregator\e[0;31m \w$]\e[m >> '\" >> ~/.bashrc && bash"


run_col_container: venv/bin/python3
	# parse the flplan to obtain model info
	$(eval module_name=$(shell venv/bin/python3 bin/flplan_info_to_stdout.py \
	  -p $(plan) -fk model -sk module_name) )
	$(eval model=$(shell echo $(module_name) | awk '{FS="." ; $$0=$$0}  { print $$4}'))
	@echo running coll container for model $(model) and collaborator col_$(col_num)

	@echo "Collaborator $(col_num) started. You are in the Docker container"
	@docker run \
	$(runtime_line) \
	--net=host \
	-it --name=tfl_col_$(device)_$(model)_$(shell whoami)_$(col_num) \
	--rm \
	-v $(shell pwd)/bin/federations:/home/$(shell whoami)/tfl/bin/federations:ro \
	$(additional_brats_container_lines) \
	-w /home/$(shell whoami)/tfl/bin \
	tfl_col_$(device)_$(model)_$(shell whoami):0.1 \
	bash -c "echo \"export PS1='\e[0;31m[FL Docker for \e[0;32mCollaborator $(col_num)\e[0;31m \w$]\e[m >> '\" >> ~/.bashrc && bash"
endif
