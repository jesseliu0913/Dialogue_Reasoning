import os
import re
import json


INPUT_FOLDER = "./input/case_report"
input_files = [f for f in os.listdir(INPUT_FOLDER) if not f.startswith('.')]
stored_pid = json.load(open("./case_seperate.json", "r"))
qualified_pid = stored_pid['qualified_pid']
empty_pid = stored_pid['empty_pid']
f_write = open("./output/tuning/raw_case2.jsonl", 'w')
pattern = r"^(A|An) \d+-year-old"
idx = 0
existing_case = []
for input_f in input_files:
  pid = input_f.split(".")[0]
  if pid not in qualified_pid and pid not in empty_pid:
    file_path = os.path.join(INPUT_FOLDER, input_f)
    file_content = json.load(open(file_path, "r"))
    for c_key in list(file_content.keys()):
      content_length = len(file_content[c_key].split("."))
      if content_length > 10 and content_length < 20 and re.match(pattern, file_content[c_key], re.IGNORECASE) and file_content[c_key] not in existing_case:
          case_content = {"pid": pid, "case": file_content[c_key]}
          existing_case.append(file_content[c_key])
          f_write.write(json.dumps(case_content) + '\n')
          idx += 1

  if idx == 10000:
     break


"""
f_write = open("./output/tuning/raw_case1.jsonl", 'w') 

For qualified case
for qid in qualified_pid:
  file_name = f"{qid}.json"
  file_path = os.path.join(INPUT_FOLDER, file_name)
  file_content = json.load(open(file_path, "r"))
  for c_key in list(file_content.keys()):
    content_length = len(file_content[c_key].split("."))
    if content_length > 10 and re.match(pattern, file_content[c_key], re.IGNORECASE):
        case_content = {"pid": qid, "case": file_content[c_key]}
        f_write.write(json.dumps(case_content) + '\n')
"""
  
