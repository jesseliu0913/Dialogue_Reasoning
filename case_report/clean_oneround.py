import os
import json


INPUT_FOLDER = './output/stage1_parse'
input_files = [f for f in os.listdir(INPUT_FOLDER) if not f.startswith(".")]
mc_files = [f for f in os.listdir('./output/stage1_cleanmc') if not f.startswith(".")]
intersec_files = list(set(input_files) - set(mc_files))

for input_f in intersec_files:
  pid = input_f.split(".")[0]
  cases = json.load(open(os.path.join(INPUT_FOLDER, input_f), "r"))
  for single_case in list(cases.keys()):
    context = cases[single_case]['oneround_dict']['input']
    question = "Could you consider all possible diseases based on my information? And then narrow down to a specific disease based on the examination? Only give me the disease name, and if there are different names for the same disease, list them separated by commas."
    answer = cases[single_case]['oneround_dict']['groundtruth']
    context = context.replace(answer, "<?>")
    if context not in ["", " ", "<?>", " <?> "]:
      case_dict = {"context": context, "question": question, "answer": answer}
      with open('./output/qualified_stage1/oneround.jsonl', 'a') as f_write:
        f_write.write(json.dumps(case_dict) + '\n')
    else:
      print(pid)

