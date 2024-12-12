import os
import sys
import json
import logging
import argparse

from load_model import *
from load_data import *
from utils import *


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--model", "-m", type=str, default="hf", help="Input the model name"
    )
    parser.add_argument(
          "--model_args",
          default="",
          type=str,
          help="Input the model args, seperate by comma",
    )
    parser.add_argument(
          "--task_name",
          default="llama",
          type=str,
          help="Input the task name",
    )
    parser.add_argument(
          "--lora_weight",
          default=None,
          type=str,
          help="Add the lora weight",
    )
    parser.add_argument(
          "--output_path",
          default="",
          type=str,
          help="Input the output folder",
    )
    parser.add_argument(
          "--confusion_level",
          default=5,
          type=int,
          help="Input the quantity of the trouble maker you want to add in the maze",
    )
    parser.add_argument(
          "--task_type",
          default="one_round",
          type=str,
          help="one_round or multi_round",
    )
    parser.add_argument(
          "--num_fewshot",
          default=0,
          type=int,
          help="Input the number of few-shots, must this function only support the one_round task now",
    )
    parser.add_argument(
          "--limit",
          default=None,
          type=int,
          help="Input the number of examples you want to test",
    )

    return parser.parse_args()


def game_start(args) -> None:
    # init files
    pretrained_name = args.model_args.split("/")[-1] if "/" in args.model_args else args.model_args
    if args.lora_weight != None:
        lora_name = args.lora_weight.split("/")[-1] if "/" in args.lora_weight else args.lora_weight
    else:
        lora_name = "None"
    folder_path = f"./{args.output_path}/{args.task_type}"
    os.makedirs(folder_path, exist_ok=True)
    
    output_file = f"./{folder_path}/{args.model}_{pretrained_name}_{lora_name}_{args.num_fewshot}_{args.confusion_level}"
    if args.limit is not None:
        output_file += f"_{args.limit}"
    output_file += ".jsonl"
    
    # load model
    model, tokenizer = init_model(model_name=args.model, model_args=args.model_args, task_name=args.task_name, lora_weight=args.lora_weight)

    # load dataset
    maze_data = MazeDatasetProcessor(confusion_level=args.confusion_level)
    if args.task_type == 'one_round':
        dataset = maze_data.get_oneround()
    elif args.task_type == 'multi_round':
        dataset = maze_data.get_multiround()
    else:
        print("Task Type must be 'one_round' or 'multi_round'")
    
    if args.limit != None:
        dataset = dataset.select(range(args.limit))

    if model == "openai":
        if args.task_type == 'one_round':
            for line in dataset:
                if args.num_fewshot == 0:
                    message = line['prompt']
                else:
                    fewshots = maze_data.get_fewshot(args.num_fewshot)
                    message = fewshots + "\n\n" + line['prompt']

                response = call_gpt(tokenizer, args.model_args, message)
                line['response'] = response

                write2json(output_file, line)
        else:
            for line in dataset:
                line_response, answer_idx, _ = multi_round(args.model_args, line, tokenizer, maze_data, openai_flag=True)
                line['response'] = line_response
                line['response_index'] = answer_idx

                write2json(output_file, line)

    else:
        if args.task_type == 'one_round':
            for line in dataset:
                if args.num_fewshot == 0:
                    message = line['prompt']
                else:
                    fewshots = maze_data.get_fewshot(args.num_fewshot)
                    message = fewshots + "\n\n" + line['prompt']

                inputs = tokenizer(message, return_tensors="pt").to('cuda')
                input_length = inputs['input_ids'].shape[1]
                with torch.no_grad():
                    response = model.generate(**inputs, max_new_tokens=40)
                line['response'] = tokenizer.decode(response[0][input_length:], skip_special_tokens=True)
                write2json(output_file, line)
        else:
            for line in dataset:
                line_response, answer_idx, _ = multi_round(model, line, tokenizer, maze_data, openai_flag=False)
                line['response'] = line_response
                line['response_index'] = answer_idx

                write2json(output_file, line)

                    
if __name__ == "__main__":
    args = get_args()
    print(args)
    game_start(args)       
