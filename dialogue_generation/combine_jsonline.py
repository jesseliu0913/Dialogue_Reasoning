import os
import json
import pickle


FOLDER_PATH = "./dialogue_set"
json_files = [f for f in os.listdir(FOLDER_PATH) if not f.startswith(".")]
output_file = open("./case_dialogue.jsonl", "a+")
trainig_pid = []

for input_f in json_files:
  f_read = open(f"{FOLDER_PATH}/{input_f}", "r")
  for line in f_read:
    output_file.write(line)
    line_pid = json.loads(line)['pid']
    trainig_pid.append(line_pid)

with open("./trainig_pid.pkl", "wb") as file:
  pickle.dump(trainig_pid, file)