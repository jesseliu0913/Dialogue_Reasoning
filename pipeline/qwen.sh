#!/bin/bash

confusion_levels=(0 3 5 10)

model="hf"
model_args="Qwen/Qwen2.5-1.5B-Instruct"
output_path="./output"
task_type="one_round"
task_name="qwen"
num_fewshot=0
save_config=""

log_dir="./log"
mkdir -p $log_dir

for confusion_level in "${confusion_levels[@]}"; do
    echo "Running with confusion_level=$confusion_level"
    
    log_file="${log_dir}/qwen25_15B_confusion_${confusion_level}.log"
    
    CUDA_VISIBLE_DEVICES=1 nohup python main.py \
        --model "$model" \
        --model_args "$model_args" \
        --output_path "$output_path" \
        --task_type "$task_type" \
        --task_name "$task_name" \
        --confusion_level $confusion_level \
        --num_fewshot $num_fewshot \
        --save_config $save_config > $log_file 2>&1

    if [ $? -ne 0 ]; then
        echo "Task failed for confusion_level=$confusion_level. Exiting."
        exit 1
    fi

    echo "Completed task with confusion_level=$confusion_level. Log saved to $log_file."
done

echo "All tasks completed successfully. Logs are available in $log_dir."

