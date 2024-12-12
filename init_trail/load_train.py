import os
import nltk
import json
import random
import numpy as np
nltk.download('punkt_tab')
random.seed(42)


TM_FLAG = False
with open("./MedQA_Maze/train.jsonl", "r") as train_file:
    train_lines = train_file.readlines() 

train_indices = random.sample(range(len(train_lines)), 2030)


if TM_FLAG == True:
  for idx, item in enumerate(train_lines[0: 2000]):
      item = json.loads(item)
      trouble_maker = item['groundtruth_zoo']
      for tm in trouble_maker:
        with open('./MedQA_Maze/trouble_maker.txt', 'a+') as f_write:
          f_write.write(tm + '\n')

  for idx, item in enumerate(train_lines[2000: 2020]):
      item = json.loads(item)
      trouble_maker = item['groundtruth_zoo']
      for tm in trouble_maker:
        with open('./MedQA_Maze/trouble_maker_shot.txt', 'a+') as f_write:
          f_write.write(tm + '\n')
else:
  print("Already have the trouble maker")

trouble_maker = open('./MedQA_Maze/trouble_maker_shot.txt', 'r').read().split("\n")
random.shuffle(trouble_maker)
trouble_maker = np.array(trouble_maker)
cl = 3

shot_dict = {}
for idx, shot_data in enumerate(train_lines[2020: 2030]):
  shot_data = json.loads(shot_data)
  choose_idx = [random.randint(0, len(trouble_maker)) for _ in range(cl)]
  choose_sentence = trouble_maker[choose_idx].tolist()
  groundtruth_zoo = shot_data['groundtruth_zoo']
  muddy_zoo = groundtruth_zoo.copy()
  trouble_indices = sorted(random.sample(range(len(groundtruth_zoo) + 1), len(choose_sentence)))

  new_trouble_indices = []
  for index, item in zip(trouble_indices, choose_sentence):
      muddy_zoo.insert(index, item)
      new_trouble_indices.append(index)
  truth_idx = [i for i in range(len(muddy_zoo)) if i not in new_trouble_indices]
  tagged_maze = "\n".join([f"{i}: {sentence}" for i, sentence in enumerate(muddy_zoo)])
  shot = f"""
Here is the background information {shot_data['prerequisit']}. 
Question: {shot_data['question']}
Answer:  {shot_data['answer']}
Below are several evidence sentences. Please identify which sentences should be added to the background information, based on the question-answer pair, to allow inference of the answer. 
{tagged_maze}
Only provide the indices of the relevant sentences, starting from index 0 and closed by [].
ANSWER: {truth_idx}
"""
  shot_dict[idx] = shot

with open("./MedQA_Maze/shot.json", "w") as f_write:
  json.dump(shot_dict, f_write)

