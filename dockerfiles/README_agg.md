How to run a TFL aggregator in a docker container.


1. Build a docker image from `Dockerfile.agg`. We only build it once unless we change `Dockerfile.agg`.
```
nvidia-docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) --build-arg UNAME=$(whoami) -t tfl:agg -f Dockerfile.agg .
```


2. Start a container in the background. Map the source code volume into the docker container in the development phase.
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


3. Get an interactive shell to the container with the command anytime.
```
docker exec -it tfl_agg /bin/bash
```


4. Install the tfedlrn package.

Currently, we install the package from the source code folder for active debugging.
```
pip install -e . --user
```

In actual deployment, we may install with a wheel.
```
pip install dist/tfedlrn-0.0.0-py3-none-any.whl  --user
```


5. Example: Build an initial model and run the aggregator with 2 collaborators.
```
python bin/build_initial_tensorflow_model.py -m TensorFlow2DUNet
python bin/simple_fl_agg.py -n 2 -i TensorFlow2DUNet --server_port 5678
```


6. Stop the cotainer running in the background after we finish everything. Goto step 2 if you need to start it later.
```
docker stop tfl_agg
```