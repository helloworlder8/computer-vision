#!/bin/bash
code ../ultra -r ultralytics/nn/my_modules/module_block.py \
ultralytics/nn/tasks_function.py \
ultralytics/cfg_yaml/test_model_yaml/ShuffleNet_24_04_04.3_lightcodattention.yaml \
script/train.py
