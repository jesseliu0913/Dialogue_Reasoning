import os
import json
import numpy as np


file_path = "/scratch0/zx22/zijie/MDAgents/data/symcat-474-symptoms/symptoms.json"
file_content = json.load(open(file_path, "r"))
key_lst = list(file_content.keys())
length_lst = []
attr_lst = ['sex', 'age', 'race', 'common_causes']

def cal_norm(prob_lst):
  min_value = min(prob_lst)
  max_value = max(prob_lst)

  if max_value == min_value:
    return False  
  else:
    return [(x - min_value) / (max_value - min_value) for x in prob_lst]

for att in attr_lst:
    for dis in key_lst[0:10]:
      org_dict = {}
      org_dict['name'] = file_content[dis]['name']

      if att ==  "common_causes" and file_content[dis]['common_causes'] != {}:
        org_dict['question'] = f"Description: {file_content[dis]['description']}\n\n"
        org_dict['common_causes'] = {}
        common_causes = file_content[dis]['common_causes']
        sorted_causes = sorted(common_causes.values(), key=lambda x: x["probability"], reverse=True)
        org_dict['common_causes']['choices'] = sorted_causes
        prob_lst = [pair['probability'] for pair in sorted_causes]
        min_value, max_value = min(prob_lst), max(prob_lst)
        normprob_lst = [(x - min_value) / (max_value - min_value) for x in prob_lst]
        org_dict['common_causes']['choices_prob'] = normprob_lst

        folder_path = f'./data/symcat_symptoms/common_causes'
        if not os.path.exists(folder_path):
           os.makedirs(folder_path)
           
        with open(f'{folder_path}/dev.jsonl', 'a+') as f:
              f.write(json.dumps(org_dict) + '\n')
         
      if att != 'common_causes' and file_content[dis][att] != {}:
        org_dict['question'] = f"Description: {file_content[dis]['description']}\n\n"
        org_dict[att] = {}
        common_causes = file_content[dis][att]
        sorted_causes = sorted(common_causes.values(), key=lambda x: x["odds"], reverse=True)
        org_dict[att]['choices'] = sorted_causes
        prob_lst = [pair['odds'] for pair in sorted_causes]
        norm_lst = cal_norm(prob_lst)
        org_dict[att]['choices_prob'] = norm_lst

        folder_path = f'./data/symcat_symptoms/{att}'
        if not os.path.exists(folder_path):
           os.makedirs(folder_path)

        with open(f'{folder_path}/dev.jsonl', 'a+') as f:
              f.write(json.dumps(org_dict) + '\n')


      # with open('./data/symcat_symptoms/test.jsonl', 'a+') as f:
      #   f.write(json.dumps(org_dict) + '\n')
