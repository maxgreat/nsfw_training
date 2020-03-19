#!/bin/bash
if [[ $1 == *"bash" ]];then
  exec /bin/bash
else
  set -e
  jupyter tensorboard eneable --user
  jupyter lab --no-browser --ip=0.0.0.0 --port=8888
fi