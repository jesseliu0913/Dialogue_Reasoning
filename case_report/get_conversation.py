import os,re
import nltk
import json
from nltk.tokenize import sent_tokenize
from DatasetTools import TextProcessingTools
from DatasetGenerator import MedicalDialogueProcessor


INPUT_FOLDER = "./output/stage1_output"
OUTPUT_FOLDER = "./output/stage1_dialogue"
GROUNDTRUTH_FOLDER = "./input/full_report"

input_files = [f for f in os.listdir(INPUT_FOLDER) if not f.startswith(".")]
groundtruth_files = [f for f in os.listdir(GROUNDTRUTH_FOLDER) if not f.startswith(".")]
output_files = [f for f in os.listdir(OUTPUT_FOLDER) if not f.startswith(".")]
question_lst = [
  "Describe the patient personal information.",
  "Describe the patient experience.",
  "Did you notice any symptoms, such as a fever, cough, or respiratory issues?",
  "What's the diagnosis?",
  "What's the direct evidence that points to this diagnosis?",
  "What's the imaging (only provided the figure explanation here) suggest?",
  "What's the examination suggest?",
  "Is there any suggestion?",
]

def generate_dialogue(single_case):
    dialogues = []
    for case_id, case_data in single_case.items():
        clean_answer = [an for an in case_data["cleaned_answer"] if an != None]
        if int(case_id) <= 2: 
          question_text = question_lst[int(case_id)]
          answer_text = " ".join(clean_answer) if clean_answer != [] else "NO"
          dialogues.append(f"Doctor: {question_text}\nPatient: {answer_text}\n")
        else:
          question_text = question_lst[int(case_id)]
          answer_text = " ".join(clean_answer) if clean_answer != [] else "NO"
          dialogues.append(f"Patient: {question_text}\nDoctor: {answer_text}\n")

    return "\n".join(dialogues)

for input_f in input_files:
  if input_f not in output_files:
      file_content = json.load(open(os.path.join(INPUT_FOLDER, input_f), "r"))
      article = json.load(open(os.path.join(GROUNDTRUTH_FOLDER, input_f), "r"))
      pid = input_f.split(".")[0]
      case_dict = {"pid": pid}
      for case_tile, single_case in file_content.items():
        dialogue = generate_dialogue(single_case)
        if dialogue != "":
          case_dict["case"] = dialogue
          with open("/Users/liuzijie/Desktop/LZJ/clinic_dg/output/stage1_dialogue/cleaned_dialogue.jsonl", "a+") as f_write:
            f_write.write(json.dumps(case_dict) + "\n")

      
    # TextProcessingTools.save_json(f'{OUTPUT_FOLDER}/{pid}.json', file_dict)

