"""
Combine all context-based dataset to create full MAZE data:
Medqa-4opts, medbulltes4op, medbulltes5op, Jama_challenge
Return Dict:
context, question, prerequisit, groundtruth_zoo, answer
"""

import nltk
import json
import random
from datasets import load_dataset
nltk.download('punkt_tab')
import argparse


dataset_repo = [
  "GBaker/MedQA-USMLE-4-options",
  "JesseLiu/medbulltes5op",
  "JesseLiu/medbulltes4op",
  "JesseLiu/Jama_challenge"
]


def write2file(ds, train=False):
  for item in ds['test']:
    item_dict = {}
    context = item['question']
    item_dict['context'] = context
    sent_context= nltk.sent_tokenize(context)
    item_dict['question'] = sent_context[-1]
    item_dict['prerequisit'] = sent_context[0]
    item_dict['groundtruth_zoo'] = sent_context[1:-1]
    item_dict['answer'] = item['answer']

    with open('MedQA_Maze/test.jsonl', 'a+') as f_write:
      f_write.write(json.dumps(item_dict) + '\n')
  
  if train == True:
    for item in ds['train']:
      item_dict = {}
      context = item['question']
      item_dict['context'] = context
      sent_context= nltk.sent_tokenize(context)
      item_dict['question'] = sent_context[-1]
      item_dict['prerequisit'] = sent_context[0]
      item_dict['groundtruth_zoo'] = sent_context[1:-1]
      item_dict['answer'] = item['answer']

      with open('MedQA_Maze/train.jsonl', 'a+') as f_write:
        f_write.write(json.dumps(item_dict) + '\n')

for dataset in dataset_repo:
  ds = load_dataset(dataset)
  if dataset == "GBaker/MedQA-USMLE-4-options":
    write2file(ds, train=True)
  else:
    write2file(ds)