whl = dist/tfedlrn-0.0.0-py3-none-any.whl
tfl = venv/lib/python3.5/site-packages/tfedlrn

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

$(whl): venv/bin/python3
	venv/bin/pip3 install setuptools
	venv/bin/pip3 install wheel
	venv/bin/python3 setup.py bdist_wheel

$(tfl): $(whl)
	venv/bin/pip3 install $(whl)

initial_models:
	mkdir initial_models

initial_models/TensorFlow2DUNet.pbuf: initial_models $(tfl)
	venv/bin/python3 bin/build_initial_tensorflow_model.py -m TensorFlow2DUNet

run_brats_unet_fed: initial_models/TensorFlow2DUNet.pbuf
	venv/bin/python3 bin/simple_fl_tensorflow_test.py -n 0 -m TensorFlow2DUNet

clean:
	rm -rf venv
	rm -rf dist
	rm -rf build
	rm -rf tfedlrn.egg-info
