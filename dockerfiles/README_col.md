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
-e CUDA_VISIBLE_DEVICES=1 \
-d --rm \
--name=tfl_col0 \
tfl:col \
bash
```


3. Get an interactive shell to the container with the command anytime.
```
docker exec -it tfl_col0 /bin/bash
```


4. Run one collabrator for example. We install the package from the source code folder for active debugging.
```shell
docker exec -it tfl_col /bin/bash
export CUDA_VISIBLE_DEVICES=1
pip install -e . --user
python bin/grpc_collaborator.py --plan_path federations/plans/mnist_a.yaml --col_id 0
```

You may enable TLS and/or mutual if you have generated the key pairs by replacing the last command with
```shell
python bin/grpc_collaborator.py --plan_path federations/plans/mnist_a.yaml --col_id 0 --enable_tls --certificate_folder files/grpc/ --require_client_auth
```


4. Stop the cotainer running in the background after we finish everything. Goto step 2 if you need to start it later.
```
docker stop tfl_col0
```