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

for full_f in full_files:
  pid = full_f.split(".")[0]
  if pid not in qualified_pid and empty_pid not in qualified_pid:
    file_path = os.path.join(FULL_FOLDER, full_f)
    file_content = json.load(open(file_path, "r"))
    print(file_content.keys())
  break