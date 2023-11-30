#!/bin/bash


docker run --gpus all -v ./model_dir:/app/model_dir --runtime nvidia -it neuradicon:1.0
