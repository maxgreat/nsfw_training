#!/bin/bash
if [[ $1 == *"bash" ]];then
  exec /bin/bash
else
  jupyter tensorboard enable --user
  jupyter lab --no-browser --ip=0.0.0.0 --port=8888 --allow-root
fi
