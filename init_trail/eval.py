import os
import json
import random
import argparse
import numpy as np
from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer


parser = argparse.ArgumentParser(description="Eval the Model")
parser.add_argument("--model_weight", type=str, required=True, help="Input the model pretrained weight")
parser.add_argument("--task_name", type=str, required=True, help="Input the task name")
parser.add_argument("--add_lora", action="store_true", help="Add lora weight")
parser.add_argument("--cl_level", type=int, help="Input the Confusion Level")

args = parser.parse_args()

print(f"Using {args.model_weight} Model")
print(f"Eval {args.task_name} Task")

if args.task_name == "medalpaca":
  tokenizer = LlamaTokenizer.from_pretrained(args.model_weight)
else:
  tokenizer = AutoTokenizer.from_pretrained(args.model_weight)

model = AutoModelForCausalLM.from_pretrained(args.model_weight, trust_remote_code="True")
if args.add_lora == True:
  model = PeftModel.from_pretrained(model, "JesseLiu/lora4combine_meditron7b")

model.to('cuda')
model.eval()

shot_content = json.load(open("./MedQA_Maze/shot.json", "r"))
few_shot = ""
for shot_key in list(shot_content.keys())[0:3]:
  few_shot += shot_content[shot_key]
  few_shot += "\n\n"


trouble_maker = open('./MedQA_Maze/trouble_maker.txt', 'r').read().split("\n")
random.shuffle(trouble_maker)
trouble_maker = np.array(trouble_maker)
confusion_level = [3, 5, 8, 12]

def generate_response(test_data, cl, trouble_maker, output_dict={}):
    choose_idx = [random.randint(0, len(trouble_maker)-1) for _ in range(cl)]
    choose_sentence = trouble_maker[choose_idx].tolist()
    groundtruth_zoo = test_data['groundtruth_zoo']
    if groundtruth_zoo != []:
        muddy_zoo = groundtruth_zoo.copy()
        if len(choose_sentence) > len(groundtruth_zoo) + 1:
            trouble_indices = sorted(random.sample(range(len(groundtruth_zoo) + 1), len(groundtruth_zoo) + 1))
        else:
            trouble_indices = sorted(random.sample(range(len(groundtruth_zoo) + 1), len(choose_sentence)))

        new_trouble_indices = []
        for index, item in zip(trouble_indices, choose_sentence):
            muddy_zoo.insert(index, item)
            new_trouble_indices.append(index)

        truth_idx = [i for i in range(len(muddy_zoo)) if i not in new_trouble_indices]
        prompt = f"""{few_shot}
Here is the background information {test_data['prerequisit']}. 
Question: {test_data['question']}
Answer:  {test_data['answer']}
Below are several evidence sentences. Please identify which sentences should be added to the background information, based on the question-answer pair, to allow inference of the answer. 
{muddy_zoo}
Only provide the indices of the relevant sentences, starting from index 0 and closed by [].
A:
"""
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to('cuda')
        output = model.generate(input_ids, max_new_tokens=100, do_sample=True, temperature=0.7)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        output_dict['prerequisit'] = test_data['prerequisit']
        output_dict['truth_idx'] = truth_idx
        output_dict['trouble_indices'] = new_trouble_indices
        output_dict['output'] = generated_text.replace(prompt, "")

    return output_dict

for cl in confusion_level:
  test_file = open("./MedQA_Maze/test.jsonl", "r")
  results_file = f'./{args.task_name}_results/confusion_{cl}.jsonl'
  if os.path.exists(results_file):
    entries = []
    with open(results_file, 'r') as f_current:
        for line in f_current:
            entries.append(json.loads(line))
    current_length = len(entries)
    if current_length == 1273:
      print(f"Already finished in confusion level {cl}")
    else:
      for idx, test_data in enumerate(test_file):
        if idx > current_length:
          test_data = json.loads(test_data)
          output_dict = generate_response(test_data, cl, trouble_maker)
          with open(f'./{args.task_name}_results/confusion_{cl}.jsonl', 'a+') as f_write:
              f_write.write(json.dumps(output_dict) + '\n')
  else:
    for test_data in test_file:
      test_data = json.loads(test_data)
      output_dict = generate_response(test_data, cl, trouble_maker)
      with open(f'./{args.task_name}_results/confusion_{cl}.jsonl', 'a+') as f_write:
          f_write.write(json.dumps(output_dict) + '\n')




