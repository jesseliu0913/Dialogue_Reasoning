import os
import json
import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

ds = load_dataset("GBaker/MedQA-USMLE-4-options")['train']
# for line in ds:
#     print(line['options'])
#     break

task_type = "eval"
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", torch_dtype=torch.float16).to("cuda")

file_path = "/scratch0/zx22/zijie/MDAgents/MuddyMaze/dialogue_medqa_noq.jsonl"

def call_hf(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=1, num_beams=5, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if response.startswith(prompt):
        response = response[len(prompt):].strip()
    return response

if task_type == "run":
  f_write = open("./0shot_eval/llama3.2_3b_dialogue.jsonl", "a")
  file = open(file_path, 'r')
  for line, ds_line in zip(file, ds):
      line_dict = {}
      data = json.loads(line.strip()) 
      dialogue = data['response'].replace("\n\n", "\n")
      question = data['question']
      options = ds_line['options']
      prompt_diag = f"{dialogue}\n{question}\n{options}\n Directly answer me without any other words. \n Answer:"
      prompt_mc = f"{ds_line['question']}\n{options}\n Directly answer me without any other words. \n Answer:"
      res_diag = call_hf(prompt_diag)
      res_mc = call_hf(prompt_mc)

      line_dict["input"] = data['input']
      line_dict["answer"] = ds_line['answer_idx']
      line_dict["res_diag"] = res_diag
      line_dict["res_mc"] = res_mc

      f_write.write(json.dumps(line_dict) + "\n")
else:
    f_read = open("./0shot_eval/llama3.2_3b_dialogue.jsonl", 'r')
    mc_score = 0
    diag_score = 0
    total_count = 0
    for data in f_read:
        line = json.loads(data.strip()) 
        total_count += 1
        answer = line["answer"]
        mc_answer = line["res_mc"]
        diag_answer = line["res_diag"]

        if answer == mc_answer:
            mc_score += 1
        if answer == diag_answer:
            diag_score += 1

print("MC score:", mc_score / total_count)
print("Diag score:", diag_score / total_count)
            
        

