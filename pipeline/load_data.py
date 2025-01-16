import os
import json
import random
from datasets import load_dataset
random.seed(42)


class MazeDatasetProcessor:
  def __init__(self, config_dict, config_flag, confusion_level=0, task_level='basic'):
      self.confusion_level = confusion_level
      self.shot_file = "./shot_data/shot.json"
      # self.trouble_maker_file = "./shot_data/trouble_maker.txt"
      self.trouble_maker_file = "./shot_data/dealed_response.jsonl"

      # self.trouble_maker = [line.strip() for line in open(self.trouble_maker_file, 'r')]
      self.trouble_maker = [json.loads(line.strip()) for line in open(self.trouble_maker_file, "r")]
      self.shots = json.load(open(self.shot_file, "r"))

      self.female_pronouns = ['she', 'her', 'hers', 'herself', 'woman']
      self.male_pronouns = ['he', 'him', 'his', 'himself', 'man']
      self.config_dict = config_dict
      self.config_flag = config_flag

      self.task_level = task_level

      # self.female_sentences, self.male_sentences, self.other_sentences = self._org_trouble_make()

   
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
  
  def read_exist_tm(self, line_idx):
     return self.config_dict[str(line_idx)]
  
  def create_maze(self, context, groundtruth_zoo, line_idx):
    if self.confusion_level != 0:
        if self.config_flag == False:
            choose_sentence = self.get_troublemaker(context, line_idx)
        else:
            choose_sentence = self.read_exist_tm(line_idx)
    
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

        combined = list(zip(muddy_zoo, truth_idx))
        random.shuffle(combined)
        muddy_zoo, truth_idx = zip(*combined)
        
        muddy_zoo = list(muddy_zoo)
        truth_idx = list(truth_idx)

        return muddy_zoo, truth_idx, trouble_index

    

  def oneround_prompt(self, line, idx):
      muddy_maze, truth_idx, new_trouble_index = self.create_maze(line['context'], line['groundtruth_zoo'], idx)
      tagged_maze = "\n".join([f"{i}: {sentence}" for i, sentence in enumerate(muddy_maze)])
      line['prompt'] = f"""Here is the background information: "{line['prerequisit']}"
Question: {line['question']}
Answer: {line['answer']}
Below are several evidence sentences. 
Identify the {len(line['groundtruth_zoo'])} sentences that, if added to the background information, would support inferring the answer based on the given question-answer pair. Please choose the sentence in logical order!
{tagged_maze}
Provide only the indices of the relevant sentences in brackets formatted like this: [ ], no more than {len(line['groundtruth_zoo'])} sentences.
ANSWER:
"""
      line['maze'] = muddy_maze
      line['truth_idx'] = truth_idx
      line['trouble_idx'] = new_trouble_index

      return line
  
  def muliround_prompt(self, line, idx):
      muddy_maze, truth_idx, new_trouble_index = self.create_maze(line['context'], line['groundtruth_zoo'], idx)
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
      dataset = load_dataset("JesseLiu/MedQA_Maze", self.task_level, split="test", trust_remote_code=True)
      dataset = dataset.map(
          lambda example, index: self.oneround_prompt(example, index),
          with_indices=True,
          load_from_cache_file=False
      )
      return dataset, self.config_dict


  def get_multiround(self):
      dataset = load_dataset("JesseLiu/MedQA_Maze", self.task_level, split="test", trust_remote_code=True)
      dataset = dataset.map(
          lambda example, index: self.muliround_prompt(example, index),
          with_indices=True,
          load_from_cache_file=False
      )
      return dataset, self.config_dict
  
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
  
  def replace_pronouns(self, context, direction="MF"):
      if direction == "MF":
          pronoun_map = dict(zip(self.male_pronouns, self.female_pronouns))
      elif direction == "FM":
          pronoun_map = dict(zip(self.female_pronouns, self.male_pronouns))
      else:
          raise ValueError("Invalid direction. Use 'MF' or 'FM'.")
      
      words = context.split()
      replaced_words = [
          pronoun_map[word.lower()] if word.lower() in pronoun_map else word
          for word in words
      ]
      
      final_words = [
          word.capitalize() if word_original[0].isupper() else word
          for word, word_original in zip(replaced_words, words)
      ]
      
      return ' '.join(final_words)
  
  def effect_dict(self, d):
    """Check if a dictionary is empty or all its values are empty lists."""
    return not d or all(isinstance(v, list) and not v for v in d.values())
  
  def get_troublemaker(self, context, line_idx):
    trouble_makers = []
    choose_idx = [random.choice([i for i in range(len(self.trouble_maker)) if i != line_idx])
              for _ in range(self.confusion_level)]

    if any(pronoun in context.split() for pronoun in self.female_pronouns):        
      for i in choose_idx:
        while self.effect_dict(self.trouble_maker[i]):
          i = random.choice([idx for idx in range(len(self.trouble_maker)) if idx not in choose_idx])
        line_keys = self.trouble_maker[i].keys()
        selected_key = random.choice([key for key in line_keys if self.trouble_maker[i][key] != []])
        selected_string = self.replace_pronouns(random.choice(self.trouble_maker[i][selected_key]), direction="MF")
        trouble_makers.append(selected_string)
    else:
      for i in choose_idx:
        while self.effect_dict(self.trouble_maker[i]):
          i = random.choice([idx for idx in range(len(self.trouble_maker)) if idx not in choose_idx])
        line_keys = self.trouble_maker[i].keys()
        selected_key = random.choice([key for key in line_keys if self.trouble_maker[i][key] != []])
        selected_string = self.replace_pronouns(random.choice(self.trouble_maker[i][selected_key]), direction="FM")
        trouble_makers.append(selected_string)
    
    self.config_dict[line_idx] = trouble_makers
    return trouble_makers

        
     