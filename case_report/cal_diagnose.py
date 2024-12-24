import os
import json


input_file = open("./output/qualified_stage1/oneround.jsonl", 'r')
diagnose_lst = []
for line in input_file:
    line_dict = json.loads(line)
    diagnose_lst.append(line_dict['answer'])
print(len(diagnose_lst))
print(len(set(diagnose_lst)))
