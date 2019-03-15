#!/usr/bin/env bash

sudo nvidia-docker run \
--net host \
--volume $1:/code \
--volume $2:/data \
--env PYTHONPATH=/code \
--workdir /code \
--publish 8888:8888 \
--publish 6006:6006 \
--publish 8080:8080 \
-d tensorflow/tensorflow:1.11.0-gpu-py3