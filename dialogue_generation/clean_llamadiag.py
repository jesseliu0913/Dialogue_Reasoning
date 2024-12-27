import os
import json
import re


f_write = open("./dialogue_set/clean_dialogue_case.jsonl", "a+")
with open('./dialogue_set/case_dialogue.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        data = json.loads(line)
        text = data['response']
        pattern = r'(Doctor|Patient): (.*?)(?=\n(?:Doctor|Patient):|\Z)'
        matches = re.findall(pattern, text, re.DOTALL)
        dialogue = [{"Speaker": match[0], "Content": match[1].strip()} for match in matches]
        clean_dialogue = []
        for idx, line in enumerate(dialogue):
            if idx == len(dialogue) - 1:
                content = line['Content'].split('.')[0]
                content = re.sub(r'\([^)]*\)', '', content)
                clean_dialogue.append(f"{line['Speaker']}: {content}\n")
            else:
                content = re.sub(r'\([^)]*\)', '', line['Content'])
                clean_dialogue.append(f"{line['Speaker']}: {content}\n")
      
        data['clean_dialogue'] = clean_dialogue
        f_write.write(json.dumps(data) + '\n')
                