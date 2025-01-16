import nltk
import json
import random
from datasets import load_dataset
# nltk.download('punkt_tab')
import argparse


dataset_repo = [
  "GBaker/MedQA-USMLE-4-options",
  "JesseLiu/medbulltes5op",
  "JesseLiu/medbulltes4op",
  "JesseLiu/Jama_challenge"
]

baisc_lst = []
advanced_lst = []
challenge_lst = []
count = 0

# basic_file = open("/playpen/xinyu/jesse/Dialogue_Reasoning/maze_generation/MedQA_Maze/basic/test.jsonl", "a+")
# advance_file = open("/playpen/xinyu/jesse/Dialogue_Reasoning/maze_generation/MedQA_Maze/advance/test.jsonl", "a+")
# challenge_file = open("/playpen/xinyu/jesse/Dialogue_Reasoning/maze_generation/MedQA_Maze/challenge/test.jsonl", "a+")

def get_line(item):
    item_dict = {}
    context = item['question']
    item_dict['context'] = context
    sent_context= nltk.sent_tokenize(context)
    item_dict['question'] = sent_context[-1]
    item_dict['prerequisit'] = sent_context[0]
    item_dict['groundtruth_zoo'] = sent_context[1:-1]
    item_dict['answer'] = item['answer']

    if item_dict['groundtruth_zoo'] != []:
      return item
    else:
      return None

# def write2file(ds, train=False):
#   for item in ds['test']:
#     item_dict = {}
#     context = item['question']
#     item_dict['context'] = context
#     sent_context= nltk.sent_tokenize(context)
#     item_dict['question'] = sent_context[-1]
#     item_dict['prerequisit'] = sent_context[0]
#     item_dict['groundtruth_zoo'] = sent_context[1:-1]
#     item_dict['answer'] = item['answer']

#     with open('MedQA_Maze/test.jsonl', 'a+') as f_write:
#       f_write.write(json.dumps(item_dict) + '\n')

# for dataset in dataset_repo:
#   ds = load_dataset(dataset)
#   if dataset == "GBaker/MedQA-USMLE-4-options":
#     for idx, line in enumerate(ds["test"]):
#       degree = line['meta_info']
#       if degree == "step2&3":
#         line_dict = get_line(line)
#         if line_dict != None:
#           advance_file.write(json.dumps(line_dict) + '\n')
#       else:
#         line_dict = get_line(line)
#         if line_dict != None:
#           basic_file.write(json.dumps(line_dict) + '\n')

#   elif dataset in ["JesseLiu/medbulltes4op", "JesseLiu/medbulltes5op"]:
#     for idx, line in enumerate(ds["test"]):
#       line_dict = get_line(line)
#       if line_dict != None:
#         advance_file.write(json.dumps(line_dict) + '\n')

#   elif dataset == "JesseLiu/Jama_challenge":
#     for idx, line in enumerate(ds["test"]):
#       line_dict = get_line(line)
#       if line_dict != None:
#         challenge_file.write(json.dumps(line_dict) + '\n')


basic_path = "/playpen/xinyu/jesse/Dialogue_Reasoning/maze_generation/MedQA_Maze/basic/test.jsonl"
advance_path = "/playpen/xinyu/jesse/Dialogue_Reasoning/maze_generation/MedQA_Maze/advance/test.jsonl"
challenge_path = "/playpen/xinyu/jesse/Dialogue_Reasoning/maze_generation/MedQA_Maze/challenge/test.jsonl"
test_path = "/playpen/xinyu/jesse/Dialogue_Reasoning/maze_generation/MedQA_Maze/all/test.jsonl"


file_lengths = {}
data = []
for file_path, label in [(basic_path, "Basic"), (advance_path, "Advance"), (challenge_path, "Challenge")]:
    count = 0
    with open(file_path, "r") as file:
        for line in file:
            data.append(json.loads(line.strip()))
            count += 1
    file_lengths[label] = count
    print(f"{label} file length: {count}")

with open(test_path, "w") as outfile:
    for item in data:
        outfile.write(json.dumps(item) + "\n")

print(f"Combined data written to {test_path}")
"""
Basic file length: 662
Advance file length: 1189
Challenge file length: 1511
"""

# medqa = load_dataset("GBaker/MedQA-USMLE-4-options")["test"]
# md4 = load_dataset("JesseLiu/medbulltes4op")["test"]
# md5 = load_dataset("JesseLiu/medbulltes5op")["test"]
# jama = load_dataset("JesseLiu/Jama_challenge")["test"]

# import pickle

# with open("./groups_info/baisc.pkl", "wb") as file:
#   pickle.dump(baisc_lst, file)

# with open("./groups_info/advance.pkl", "wb") as file:
#   pickle.dump(advanced_lst, file)

# with open("./groups_info/challenge.pkl", "wb") as file:
#   pickle.dump(challenge_lst, file)

