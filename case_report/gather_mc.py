import os
import re
import json
import string
import random


INPUT_FOLDER = "./output/stage1_mc"
DIAG_FOLDER = "./output/stage1_parse/"
OUTPUT_FOLDER = "./output/stage1_cleanmc"
input_files = [f for f in os.listdir(INPUT_FOLDER) if not f.startswith('.')]


def extract_information(text, ruled_out, comfirmed):
    pattern = r'(?P<label>Diagnosis ruled out|Reason for exclusion|Confirmed diagnosis|Reason for confirmation):\s*(?P<content>.*?)(?=(Diagnosis ruled out|Reason for exclusion|Confirmed diagnosis|Reason for confirmation|$))'
    matches = re.finditer(pattern, text, re.DOTALL)
    extracted_info = []
    for match in matches:
        label = match.group('label')
        # print(label)
        if "ruled" in label or "exclusion" in label:
          content = match.group('content').strip()
          ruled_out.append(content)
        elif "Confirmed" in label or "confirmation" in label:
          content = match.group('content').strip()
          comfirmed.append(content)
    return ruled_out, comfirmed


def get_case_dict(input_f):
    all_cases = [json.loads(line) for line in open(os.path.join(INPUT_FOLDER, input_f), "r")]
    cases_mc = []
    for case_content in all_cases:
        case_mc = {}
        ruled_out = []
        comfirmed = []
        response1 = case_content['response1']
        response2 = case_content['response2']

        ruled_out, comfirmed = extract_information(response1, ruled_out, comfirmed)
        ruled_out, comfirmed = extract_information(response2, ruled_out, comfirmed)

        case_mc['ruled_out'] = ruled_out
        case_mc['comfirmed'] = comfirmed

        cases_mc.append(case_mc)
    return cases_mc

for input_f in input_files:
    pid = input_f.split(".")[0]
    print(pid)
    if pid != "9019895":
        cases_mc = get_case_dict(input_f)[0]
        patient_info = json.load(open(os.path.join(DIAG_FOLDER, f"{pid}.json"), "r"))
        if patient_info != {} and cases_mc['ruled_out'] != []:
            case_bg = patient_info[list(patient_info.keys())[0]]['oneround_dict']
            if case_bg['input'] != "":
                choice_lst = cases_mc['ruled_out'][::2]
                correct_answer = cases_mc['comfirmed'][0]
                choice_lst.append(correct_answer)
                random.shuffle(choice_lst)

                alphabet_keys = list(string.ascii_uppercase[:len(choice_lst)])
                choice_dict = {alphabet_keys[i]: choice_lst[i] for i in range(len(choice_lst))}
                reversed_dict = {v: k for k, v in choice_dict.items()}
                correct_choice = reversed_dict.get(correct_answer)
                
                choice_string = "\n".join([f"{key}: {value}" for key, value in choice_dict.items()])
                mc_dict = {"context": case_bg['input'], "question": f"What is the most likely diagnosis?\n{choice_string}", "answer":correct_choice}

                with open(os.path.join(OUTPUT_FOLDER, input_f), 'w') as json_file:
                    json.dump(mc_dict, json_file, indent=4)
                
                with open("./output/qualified_stage1/mc.jsonl", 'a+') as file:
                    file.write(json.dumps(mc_dict) + '\n')