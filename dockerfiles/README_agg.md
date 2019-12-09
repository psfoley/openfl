# Tutorial: How to run a TFL aggregator in a docker container.

1. Enter the project folder and clean the build folder.
```shell
cd spr_secure_intelligence-trusted_federated_learning
make clean
```

2. Build a docker image from `Dockerfile.agg`.
We only build it once unless we change `Dockerfile.agg`.
We create a user with the same UID so that it is easier to access local volume from the docker container.
```shell
docker build \
  --build-arg UID=$(id -u) \
  --build-arg GID=$(id -g) \
  --build-arg UNAME=$(whoami) \
  -t tfl_agg:0.1 \
  -f ./dockerfiles/Dockerfile.agg \
  .
```

3. Create several aliases to simplify the docker usage.
First, we create an alias to run the docker container.
We map the local volume `./bin/` to the docker container.
```shell
alias tfl-agg-docker='docker run \
  --net=host \
  -it --name=tfl_agg \
  --rm \
  -v "$PWD"/bin:/home/$(whoami)/tfl/bin:rw \
  -w /home/$(whoami)/tfl/bin \
  tfl_agg:0.1'
```
Second, we create an alias to make the certificates that are required by TLS.
```shell
alias tfl-make-local-certs='tfl-agg-docker bash -c "cd ..; make local_certs"'
```
Third, we create an alias to run aggregators.
```shell
alias tfl-aggregator='tfl-agg-docker \
  ../venv/bin/python3 run_aggregator_from_flplan.py'
```

4. Generate the certificates for TLS communication.
```shell
tfl-make-local-certs
```

5. Start an aggregator. 
```shell
tfl-aggregator -p mnist_a.yaml
```

In case anytime you need to examine the docker container with a shell, just type
```shell
tfl-agg-docker bash
```
