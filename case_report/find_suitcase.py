import os
import json


PARSE_FOLDER = "./output/stage1_parse"
parsed_files = [f for f in os.listdir(PARSE_FOLDER) if not f.startswith('.')]
empty_pid = []
qualified_pid = []

for idx, input_f in enumerate(parsed_files):
  pid = input_f.split(".")[0]
  input_path = os.path.join(PARSE_FOLDER, input_f)
  file_content = json.load(open(os.path.join(PARSE_FOLDER, input_f), "r"))
  if file_content == {}:
    empty_pid.append(pid)
  else:
    FLAG = True
    for f_key in list(file_content.keys()):
      case = file_content[f_key]
      for i in range(8):
        content = case[str(i)]
        if content['answer'] == [] or content['answer'] == ['  ']:
          FLAG = False
          continue
    if FLAG == True:
      qualified_pid.append(pid)
    else:
      empty_pid.append(pid)

case_seperate = {"qualified_pid": qualified_pid, "empty_pid": empty_pid}
with open("case_seperate.json", "w") as f_write:
  json.dump(case_seperate, f_write, indent=4)
