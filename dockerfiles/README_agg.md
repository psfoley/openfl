1. Build an docker image.
```
nvidia-docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) --build-arg UNAME=$(whoami) -t tfl:agg -f Dockerfile.agg .
```

2. Start a container in the background.
```
nvidia-docker run -it --net=host \
-w /tfl/ \
-v /home/weilinxu/coder/spr_secure_intelligence-trusted_federated_learning:/tfl \
-e CUDA_VISIBLE_DEVICES=0 \
-d --rm \
--name=tfl_agg \
tfl:agg \
bash
```



3. Enter a container.
```
docker exec -it tfl_agg /bin/bash
```

4. Stop a cotainer running in the background.
```
docker stop tfl_agg
```



In aggerator:
```
pip install dist/tfedlrn-0.0.0-py3-none-any.whl  --user
python bin/build_initial_tensorflow_model.py -m TensorFlow2DUNet
python bin/simple_fl_agg.py -n 1 -i TensorFlow2DUNet --server_port 5678
```

Under development, we should use the active folder as the package path.
```
pip install -e . --user
```
