import os
import json
import pickle


OUTPUT_FOLDER = "../output"
oneround_files = [f for f in os.listdir(f"{OUTPUT_FOLDER}/all/one_round") if not f.startswith(".")]
multiround_files = [f for f in os.listdir(f"{OUTPUT_FOLDER}/all/multi_round") if not f.startswith(".")]

group_types = ['basic', 'advance', 'challenge']

with open("../groups_info/basic.pkl", "rb") as file:
    basic_lst = pickle.load(file)

with open("../groups_info/advance.pkl", "rb") as file:
    advance_lst = pickle.load(file)

with open("../groups_info/challenge.pkl", "rb") as file:
    challenge_lst = pickle.load(file)

for gt in group_types:
  folder_path = os.path.join(OUTPUT_FOLDER, gt)
  if not os.path.exists(folder_path):
      os.makedirs(folder_path)
  
  if not os.path.exists(f"{folder_path}/one_round"):
      os.makedirs(f"{folder_path}/one_round")
  
  if not os.path.exists(f"{folder_path}/multi_round"):
      os.makedirs(f"{folder_path}/multi_round")

for multi_f in multiround_files:
    multi_file = open(f"{OUTPUT_FOLDER}/all/multi_round/{multi_f}", "r")
    basic_file_path = f"{OUTPUT_FOLDER}/basic/multi_round/{multi_f}"
    
    if os.path.exists(basic_file_path):
        print("File exists, Skiping!")
        continue

    basic_file = open(f"{OUTPUT_FOLDER}/basic/multi_round/{multi_f}", "a+")
    advance_file = open(f"{OUTPUT_FOLDER}/advance/multi_round/{multi_f}", "a")
    challenge_file = open(f"{OUTPUT_FOLDER}/challenge/multi_round/{multi_f}", "a")

    for idx, line in enumerate(multi_file):
        if idx in basic_lst:
            basic_file.write(line)
        elif idx in advance_lst:
            advance_file.write(line)
        elif idx in challenge_lst:
            challenge_file.write(line)
            
for one_f in oneround_files:
    one_file = open(f"{OUTPUT_FOLDER}/all/one_round/{one_f}", "r")
    basic_file_path = f"{OUTPUT_FOLDER}/basic/one_round/{one_f}"
    
    if os.path.exists(basic_file_path):
        print("File exists, Skiping!")
        continue
    else:
        print(basic_file_path)
    
    basic_file = open(f"{OUTPUT_FOLDER}/basic/one_round/{one_f}", "a+")
    advance_file = open(f"{OUTPUT_FOLDER}/advance/one_round/{one_f}", "a")
    challenge_file = open(f"{OUTPUT_FOLDER}/challenge/one_round/{one_f}", "a")

    for idx, line in enumerate(one_file):
        if idx in basic_lst:
            basic_file.write(line)
        elif idx in advance_lst:
            advance_file.write(line)
        elif idx in challenge_lst:
            challenge_file.write(line)

    




