import os
import re
import json
from bs4 import BeautifulSoup


INPUT_FOLDER = "./input/case_report"
FULL_FOLDER = "./input/PMC_patient_data"
input_files = [f for f in os.listdir(INPUT_FOLDER) if not f.startswith('.')]
full_files = [f for f in os.listdir(FULL_FOLDER) if not f.startswith('.')]
stored_pid = json.load(open("./case_seperate.json", "r"))
qualified_pid = stored_pid['qualified_pid']
empty_pid = stored_pid['empty_pid']
f_write = open("./output/tuning/rare_case.jsonl", 'a+')
count = 0
for full_f in full_files:
  pid = full_f.split(".")[0]
  file_path = os.path.join(FULL_FOLDER, full_f)
  with open(file_path, 'r', encoding='utf-8') as file:
    xml_content = file.read()
  soup = BeautifulSoup(xml_content, 'xml')
  title = soup.find('article-title')
  if "Rare Case" in title.text:
    count += 1
    if count > 1000:
      case_content = {"pid": pid, "title": title.text}
      f_write.write(json.dumps(case_content) + '\n')
  if count == 5000:
    break
