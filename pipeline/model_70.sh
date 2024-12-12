#!/bin/bash

CUDA_VISIBLE_DEVICES=0 nohup python main.py \
--model "hf" \
--model_args "meta-llama/Llama-3.1-70B-Instruct" \
--output_path "./output" \
--task_type "multi_round" \
--task_name "llama" \
--confusion_level 3 > ./log/llama32.70b 2>&1 &
