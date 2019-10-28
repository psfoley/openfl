How to run a TFL aggregator in a docker container.


1. Build a docker image from `Dockerfile.agg`. We only build it once unless we change `Dockerfile.agg`.
```shell
nvidia-docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) --build-arg UNAME=$(whoami) -t tfl:agg -f Dockerfile.agg .
```


2. Start a container in the background. Map the source code folder into the docker container in the development phase. We don't need a GPU for the aggregator since all operations are implemented with NumPy.
```shell
nvidia-docker run -it \
--net=host \
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


5. Run an MNIST example with only one collaborator.
```shell
python bin/grpc_aggregator.py --plan_path federations/plans/mnist_a.yaml
```

You can enable TLS and/or mutual authentication if you have generated the key pairs.
```shell
python bin/grpc_aggregator.py --plan_path federations/plans/mnist_a.yaml --enable_tls --certificate_folder files/grpc/ --require_client_auth
```

Now switch to `README_col.md` to start a collaborator.


6. You may stop the cotainer running in the background after we finish everything. Goto step 2 if you need to start it later.
```
docker stop tfl_agg
```
