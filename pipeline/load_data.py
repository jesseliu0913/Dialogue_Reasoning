import os
import json
import random
from datasets import load_dataset



class MazeDatasetProcessor:
  def __init__(self, confusion_level: int = 8):
      self.confusion_level = confusion_level
      self.shot_file = "./shot_data/shot.json"
      self.trouble_maker_file = "./shot_data/trouble_maker.txt"

      self.trouble_maker = [line.strip() for line in open(self.trouble_maker_file, 'r')]
      self.shots = json.load(open(self.shot_file, "r"))

      self.female_pronouns = ['she', 'her', 'hers', 'herself', 'woman']
      self.male_pronouns = ['he', 'him', 'his', 'himself', 'man']

      self.female_sentences, self.male_sentences, self.other_sentences = self._org_trouble_make()

   
  def _org_trouble_make(self):
      female_sentences = []
      male_sentences = []
      other_sentences = []

      for sentence in self.trouble_maker:
          sentence_lower = sentence.lower() 
          if any(pronoun in sentence_lower for pronoun in self.female_pronouns):
              female_sentences.append(sentence)
          elif any(pronoun in sentence_lower for pronoun in self.male_pronouns):
              male_sentences.append(sentence)
          else:
              other_sentences.append(sentence)
      
      return female_sentences, male_sentences, other_sentences
  
  def create_maze(self, context, groundtruth_zoo):
    if any(answer in context for answer in self.female_pronouns):
       trouble_maker = self.female_sentences + self.other_sentences
    else:
       trouble_maker = self.male_sentences + self.other_sentences
    
    if self.confusion_level != 0:
        choose_idx = [random.randint(0, len(trouble_maker) - 1) for _ in range(self.confusion_level)]
        choose_sentence = [trouble_maker[i] for i in choose_idx]

        if len(groundtruth_zoo) > len(choose_sentence):
          muddy_zoo = groundtruth_zoo.copy()
          trouble_index = sorted(random.sample(range(len(groundtruth_zoo) + 1), len(choose_sentence)))

          for index, item in zip(trouble_index, choose_sentence):
              muddy_zoo.insert(index, item)

          truth_idx = [i for i in range(len(muddy_zoo)) if i not in trouble_index]

        else:
          muddy_zoo = choose_sentence.copy()
          truth_idx = sorted(random.sample(range(len(choose_sentence) + 1), len(groundtruth_zoo)))
          new_truth_index = []
          for index, item in zip(truth_idx, groundtruth_zoo):
              muddy_zoo.insert(index, item)

          trouble_index = [i for i in range(len(muddy_zoo)) if i not in new_truth_index]
        
        return muddy_zoo, truth_idx, trouble_index
    else:
        muddy_zoo = groundtruth_zoo.copy()
        trouble_index = []
        truth_idx = [i for i in range(len(muddy_zoo))]
        return muddy_zoo, truth_idx, trouble_index

    

  def oneround_prompt(self, line):
      muddy_maze, truth_idx, new_trouble_index = self.create_maze(line['context'], line['groundtruth_zoo'])
      tagged_maze = "\n".join([f"{i}: {sentence}" for i, sentence in enumerate(muddy_maze)])
      line['prompt'] = f"""Here is the background information: "{line['prerequisit']}"
Question: {line['question']}
Answer: {line['answer']}
Below are several evidence sentences. 
Identify the {len(line['groundtruth_zoo'])} sentences that, if added to the background information, would support inferring the answer based on the given question-answer pair.
{tagged_maze}
Provide only the indices of the relevant sentences in brackets formatted like this: [ ], no more than {len(line['groundtruth_zoo'])} sentences.
ANSWER:
"""
      line['maze'] = muddy_maze
      line['truth_idx'] = truth_idx
      line['trouble_idx'] = new_trouble_index

      return line
  
  def muliround_prompt(self, line):
      muddy_maze, truth_idx, new_trouble_index = self.create_maze(line['context'], line['groundtruth_zoo'])
      tagged_maze = "\n".join([f"{i}: {sentence}" for i, sentence in enumerate(muddy_maze)])
      line['prompt'] = f"""Here is the background information: "{line['prerequisit']}"
Question: {line['question']}
Answer: {line['answer']}
Below are several evidence sentences. Based on the given question-answer pair, please select which sentence should be added to the background information to support inference of the answer. 
{tagged_maze}
You have {len(line['groundtruth_zoo'])} attempts in total to make a selection; this is your first attempt. Please choose the sentence in logical order!
Provide only the indices of the relevant sentences in brackets formatted like this: [ ]
ANSWER:"""
      line['maze'] = muddy_maze
      line['truth_idx'] = truth_idx
      line['trouble_idx'] = new_trouble_index

      return line

  def get_oneround(self):
    dataset = load_dataset("JesseLiu/MedQA_Maze", split="test")
    return dataset.map(self.oneround_prompt, load_from_cache_file=False)

  def get_multiround(self):
    dataset = load_dataset("JesseLiu/MedQA_Maze", split="test")
    return dataset.map(self.muliround_prompt, load_from_cache_file=False)
  
  def get_fewshot(self, fewshot_num=3):
    extract_shots = list(self.shots.values())[:fewshot_num]
    return "\n\n".join(map(str, extract_shots))
  
  def get_roundprompt(self, round_num, line, answer_sentence, updated_maze):
    tagged_maze = "\n".join([f"{i}: {sentence}" for i, sentence in enumerate(updated_maze)])
    prompt = f"""In the last time, you choose the sentence: "{answer_sentence}"
Here is the updated background information: "{line['prerequisit'] + answer_sentence}" 
Question: {line['question']}
Answer: {line['answer']}
Below are several evidence sentences. Based on the given question-answer pair, please select which sentence should be added to the background information to support inference of the answer. 
{tagged_maze}
You have {len(line['groundtruth_zoo'])} attempts in total to make a selection; this is your {round_num} attempt. Please choose the sentence in logical order!
Provide only the indices of the relevant sentences in brackets formatted like this: [ ]
ANSWER:"""
    return prompt