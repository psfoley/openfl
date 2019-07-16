1. Build an docker image.
```
nvidia-docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) --build-arg UNAME=$(whoami) -t tfl:col -f Dockerfile.col .
```

2. Start a container in the background.
```
nvidia-docker run -it --net=host \
-w /tfl/ \
-v /home/weilinxu/coder/spr_secure_intelligence-trusted_federated_learning:/tfl \
-v /raid/datasets/BraTS17:/raid/datasets/BraTS17 \
-e CUDA_VISIBLE_DEVICES=1 \
-d --rm \
--name=tfl_col1 \
tfl:col \
bash
```



3. Enter a container.
```
docker exec -it tfl_col1 /bin/bash
```

4. Stop a cotainer running in the background.
```
docker stop tfl_col1
```


In collabrator:
```
pip install dist/tfedlrn-0.0.0-py3-none-any.whl  --user
python bin/simple_fl_tensorflow_col.py --col_num 0 --num_collaborators 1 --model_id TensorFlow2DUNet --server_addr 127.0.0.1 --server_port 5678
```
