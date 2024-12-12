import nltk
import json
import random
from datasets import load_dataset
nltk.download('punkt_tab')
import argparse


# parser = argparse.ArgumentParser(description="Add the datatset name you ")
# parser.add_argument('--data_name', type=str, help='Input the data name')
# args = parser.parse_args()

# data_name = args.data_name

dataset_repo = [
  "GBaker/MedQA-USMLE-4-options",
  "JesseLiu/medbulltes5op",
  "JesseLiu/medbulltes4op",
  "JesseLiu/Jama_challenge"
]

for dataset in dataset_repo:
  ds = load_dataset(dataset)
  write2file(ds)


def write2file(ds):
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


