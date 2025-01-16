CUDA_VISIBLE_DEVICES=0 nohup python main.py --model "hf" --model_args "meta-llama/Llama-3.2-3B-Instruct" --output_path "./output" --task_type "one_round" --task_name 'llama' --confusion_level 0 --num_fewshot 0 --config 'cl_0' > ./log/llama32_raw0.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python main.py --model "hf" --model_args "meta-llama/Llama-3.2-3B-Instruct" --output_path "./output" --task_type "multi_round" --task_name 'llama' --confusion_level 0 --num_fewshot 0 --config 'cl_0' > ./log/llama32_raw0m.log 2>&1 &

