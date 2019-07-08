whl = dist/tfedlrn-0.0.0-py3-none-any.whl
tfl = venv/lib/python3.6/site-packages/tfedlrn

.PHONY: wheel
wheel: $(whl)

.PHONY: install
install: $(tfl)

venv/bin/python3:
	python3 -m venv venv        

$(whl): venv/bin/python3
	venv/bin/pip3 install setuptools
	venv/bin/pip3 install wheel
	venv/bin/python3 setup.py bdist_wheel

$(tfl): $(whl)
	venv/bin/pip3 install $(whl)

initial_models:
	mkdir initial_models

initial_brats_unet: initial_models
	venv/bin/python3 bin/build_initial_tensorflow_model.py -m TensorFlow2DUNet

clean:
	venv/bin/python3 setup.py clean
	rm -rf venv
	rm -rf dist
	rm -rf build
	rm -rf tfedlrn.egg-info
