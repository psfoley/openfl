tfedlrn/dist/tfedlrn-0.0.0-py3-none-any.whl: venv/bin/python3
	venv/bin/pip3 install setuptools
	venv/bin/pip3 install wheel
    venv/bin/python3 setup.py bdist_wheel

venv/bin/python3:
    python3 -m venv venv        

install: tfedlrn/dist/tfedlrn-0.0.0-py3-none-any.whl
    venv/bin/pip3 install tfedlrn/dist/tfedlrn-0.0.0-py3-none-any.whl
        ./python3.manifest.sgx scripts/test.py

run_agg: $(manifests).sgx venv/bin/python3
        ./python3.manifest.sgx scripts/simple_fl_agg.py -n 1 -i PyTorchMNISTCNN

train_mnist: $(manifests).sgx venv/bin/python3
        ./python3.manifest.sgx scripts/train_mnist.py

clean-tmp:
        rm -f python3.manifest.sgx
        rm -rf venv
