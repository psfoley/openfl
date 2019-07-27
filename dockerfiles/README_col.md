How to run TFL collaborators on a docker container.


1. Build a docker image from `Dockerfile.col`. We only build it once unless we change `Dockerfile.col`.
```
nvidia-docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) --build-arg UNAME=$(whoami) -t tfl:col -f Dockerfile.col .
```


2. Start a container in the background. Map the source code volume and the dataset volume in the development phase. We only need one running container even if we want to simulate multiple collaborators on one host.
```
nvidia-docker run -it --net=host \
-w /tfl/ \
-v /home/weilinxu/coder/spr_secure_intelligence-trusted_federated_learning:/tfl \
-v /raid/datasets/BraTS17:/raid/datasets/BraTS17 \
-e CUDA_VISIBLE_DEVICES=1 \
-d --rm \
--name=tfl_col \
tfl:col \
bash
```


3. Get an interactive shell to the container with the command anytime.
```
docker exec -it tfl_col /bin/bash
```


4. Run two collabrators for example. We install the package from the source code folder for active debugging.
```
docker exec -it tfl_col /bin/bash
export CUDA_VISIBLE_DEVICES=5
pip install -e . --user
python bin/simple_fl_tensorflow_col.py --col_num 0 --num_collaborators 2 --model_id TensorFlow2DUNet --server_addr 127.0.0.1 --server_port 5678 --opt_treatment RESET
```

```
docker exec -it tfl_col /bin/bash
export CUDA_VISIBLE_DEVICES=6
pip install -e . --user
python bin/simple_fl_tensorflow_col.py --col_num 1 --num_collaborators 2 --model_id TensorFlow2DUNet --server_addr 127.0.0.1 --server_port 5678 --opt_treatment RESET
```


4. Stop the cotainer running in the background after we finish everything. Goto step 2 if you need to start it later.
```
docker stop tfl_col
```
