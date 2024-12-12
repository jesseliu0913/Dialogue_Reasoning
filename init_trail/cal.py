import os
import re
import json
import random
import argparse
import numpy as np


task_name = ['biomistral_results', 'llama_results', 'meditron_results', 'medition_lora_results', 'medalpaca_results']
for tn in task_name:
  print(f"Current Task is {tn}")
  result_files = [f for f in os.listdir(f"./{tn}") if not f.startswith(".")]
  for result_f in result_files:
    test_file = open(f"./{tn}/{result_f}", "r")
    total_score = 0
    count = 0
    for idx, test_data in enumerate(test_file):
      test_data = json.loads(test_data)
      output = test_data['output'].split("\n\n")[0].strip()
      numbers = re.findall(r'\d+', output)
      if numbers:
        numbers = [int(num) for num in numbers]
        truth_idx = test_data['truth_idx']
        intersec = list(set(numbers) & set(truth_idx))
        if len(truth_idx) == 0:
          pass
        else:
          score = len(intersec) / len(truth_idx)
          total_score += score
          count += 1
      else:
        score = 0
        total_score += score
        count += 1
    print(f"Total Score for confusion level {result_f} is", total_score / count)

