import os
import json



# fileA_path = "/scratch0/zx22/zijie/MDAgents/lm-evaluation-harness/output/meta-llama__Llama-2-7b-chat-hf/samples_medexplain_2024-11-06T16-29-20.277353.jsonl"
# fileA_content = open(fileA_path, "r")
# fileB_path = "/scratch0/zx22/zijie/MDAgents/lm-evaluation-harness/output/meta-llama__Llama-2-7b-chat-hf/samples_medexplain_2024-11-06T16-29-30.237352.jsonl"
# fileB_content = open(fileB_path, "r")


# for idx, (lineA, lineB) in enumerate(zip(fileA_content, fileB_content)):
#   dataA = json.loads(lineA)
#   dataB = json.loads(lineB)

#   print(dataA['target'])
#   print(dataA['filtered_resps'])
#   print(dataB['filtered_resps'])
#   break

file_path = "/scratch0/zx22/zijie/MDAgents/lm-evaluation-harness/output/meta-llama__Llama-2-7b-chat-hf/samples_medexplain4op_2024-11-06T23-12-13.482770.jsonl"
file_content = open(file_path, "r")

for item in file_content:
  data = json.loads(item)
  print(data.keys())
  print(data['prompt_hash'])
  break