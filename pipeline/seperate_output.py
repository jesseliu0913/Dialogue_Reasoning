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
test_path = "/playpen/xinyu/jesse/Dialogue_Reasoning/maze_generation/MedQA_Maze/all/test.jsonl"

baisc_lst = []
advanced_lst = []
challenge_lst = []
count = 0

for dataset in dataset_repo:
  ds = load_dataset(dataset)
  if dataset == "GBaker/MedQA-USMLE-4-options":
    for idx, line in enumerate(ds["test"]):
      degree = line['meta_info']
      if degree == "step2&3":
        advanced_lst.append(count)
      else:
        baisc_lst.append(count)
      count += 1
  elif dataset in ["JesseLiu/medbulltes4op", "JesseLiu/medbulltes5op"]:
    for idx, line in enumerate(ds["test"]):
      advanced_lst.append(count)
      count += 1
  elif dataset == "JesseLiu/Jama_challenge":
    for idx, line in enumerate(ds["test"]):
      challenge_lst.append(count)
      count += 1

basic_file = open("/playpen/xinyu/jesse/Dialogue_Reasoning/maze_generation/MedQA_Maze/basic/test.jsonl", "a+")
advance_file = open("/playpen/xinyu/jesse/Dialogue_Reasoning/maze_generation/MedQA_Maze/advance/test.jsonl", "a+")
challenge_file = open("/playpen/xinyu/jesse/Dialogue_Reasoning/maze_generation/MedQA_Maze/challenge/test.jsonl", "a+")

with open(test_path, "r") as f_read:
  for idx, line in enumerate(f_read):
    if idx in baisc_lst:
      basic_file.write(line)
    elif idx in advanced_lst:
      advance_file.write(line)
    elif idx in challenge_lst:
      challenge_file.write(line)

# print("advanced_lst", len(advanced_lst))
# print("baisc_lst", len(baisc_lst))
# print("challenge_lst", len(challenge_lst))

# medqa = load_dataset("GBaker/MedQA-USMLE-4-options")["test"]
# md4 = load_dataset("JesseLiu/medbulltes4op")["test"]
# md5 = load_dataset("JesseLiu/medbulltes5op")["test"]
# jama = load_dataset("JesseLiu/Jama_challenge")["test"]

import pickle

with open("./groups_info/baisc.pkl", "wb") as file:
  pickle.dump(baisc_lst, file)

with open("./groups_info/advance.pkl", "wb") as file:
  pickle.dump(advanced_lst, file)

with open("./groups_info/challenge.pkl", "wb") as file:
  pickle.dump(challenge_lst, file)

