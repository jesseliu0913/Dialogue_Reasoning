import os
import re
import torch
import json
from load_model import *
from load_data import *


def write2json(filename, line):
  with open(filename, 'a+') as file:
      json.dump(line, file)
      file.write('\n')


def find_integer(context):
  match = re.search(r'\d+', context)
  return int(match.group()) if match else None
        

def multi_round(model, line, tokenizer, maze_data, openai_flag):
    line_response = []
    answer_idx = []
    answer_sentence = []
    original_idx_lst = []

    # start with the round 0
    init_prompt = line['prompt']
    # print("init_prompt", init_prompt)
    response = call_model(tokenizer, model, init_prompt, max_new_tokens=1, openai=openai_flag)
    line_response.append(response)
    
    # now start the remaining rounds
    original_maze_idx = {sentence: idx for idx, sentence in enumerate(line['maze'])}
    updated_prompt = init_prompt + response
    updated_maze = line['maze']

    last_answer_idx = find_integer(response)
    answer_idx.append(last_answer_idx)
    if last_answer_idx is not None and last_answer_idx < len(updated_maze):
      last_answer = updated_maze[int(last_answer_idx)] 
      answer_sentence.append(last_answer)
      original_idx = original_maze_idx.get(last_answer, None)
      original_idx_lst.append(original_idx)
    else:
      last_answer = response
      original_idx_lst.append("$")
    
    for round_num in range(1, len(line['groundtruth_zoo'])):
      if last_answer_idx is not None and last_answer_idx < len(updated_maze):
          updated_maze.pop(last_answer_idx)
                                                              
      round_prompt = maze_data.get_roundprompt(round_num, line, last_answer, updated_maze)
      # print("round_prompt", round_prompt)
      # message = updated_prompt + round_prompt
      message = round_prompt
      response = call_model(tokenizer, model, message, max_new_tokens=10, openai=openai_flag)
      # print("response", response)
      line_response.append(response)

      # updated_prompt += f"\n{round_prompt}{response}"
  
      last_answer_idx = find_integer(response)
      answer_idx.append(last_answer_idx)
      if last_answer_idx is not None and last_answer_idx < len(updated_maze):
        last_answer = updated_maze[int(last_answer_idx)] 
        answer_sentence.append(last_answer)
        original_idx = original_maze_idx.get(last_answer, None)
        original_idx_lst.append(original_idx)
      else:
        last_answer = response
        original_idx_lst.append("$")
    
    return line_response, original_idx_lst, answer_sentence