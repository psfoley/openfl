# Tutorial: How to run TFL collaborators on a docker container.

We build the Docker image for collaborators upon the aggregator image, adding necessary dependencies such as the mainstream deep learning frameworks. You may modify `./dockerfiles/Dockerfile.col` to install the needed packages.

1. Enter the project folder and clean the build folder.
```shell
cd spr_secure_intelligence-trusted_federated_learning
make clean
```

2. Build the aggregator image, which is the parent of the collaborator image (`Dockerfile.agg`).
```shell
docker build \
  --build-arg http_proxy \
  --build-arg https_proxy \
  --build-arg socks_proxy \
  --build-arg ftp_proxy \
  --build-arg no_proxy \
  --build-arg UID=$(id -u) \
  --build-arg GID=$(id -g) \
  --build-arg UNAME=$(whoami) \
  -t tfl_agg:0.1 \
  -f ./dockerfiles/Dockerfile.agg \
  .
```

3. Build a docker image from `Dockerfile.col`. We only build it once unless we change `Dockerfile.col`.
```
docker build \
  -t tfl_col:0.1 \
  -f ./dockerfiles/Dockerfile.col \
  .
```


4. Create several aliases to simplify the docker usage.
First, we create an alias to run the docker container.
We map the local volumes `./models/` and `./bin/` to the docker container.
```shell
alias tfl-col-docker='docker run \
  --net=host \
  -it --name=tfl_col \
  --rm \
  -v "$PWD"/models:/home/$(whoami)/tfl/models:ro \
  -v "$PWD"/bin:/home/$(whoami)/tfl/bin:rw \
  -w /home/$(whoami)/tfl/bin \
  tfl_col:0.1'
```
Second, we create an alias to run collaborators.
```shell
alias tfl-collaborator='tfl-col-docker \
  ../venv/bin/python3 run_collaborator_from_flplan.py'
```


5. Start a collaborator. 
```shell
tfl-collaborator -p mnist_a.yaml -col col_0
```


In case anytime you need to examine the docker container with a shell, just type
```shell
tfl-col-docker bash
```

