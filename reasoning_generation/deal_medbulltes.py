import json

file_content = json.load(open("/scratch0/zx22/zijie/MDAgents/data/medbullets_op5.json", "r"))
keys = list(file_content.keys())
num_cases = len(file_content['question'])
print(keys)
for idx in range(num_cases-10, num_cases):
  org_dict= {}
  org_dict['question'] = file_content['question'][f'{idx}']
  org_dict['choicesA'] = file_content['opa'][f'{idx}']
  org_dict['choicesB'] = file_content['opb'][f'{idx}']
  org_dict['choicesC'] = file_content['opc'][f'{idx}']
  org_dict['choicesD'] = file_content['opd'][f'{idx}']
  org_dict['choicesE'] = file_content['ope'][f'{idx}']
  # org_dict['choices'] = f"A. {file_content['opa'][f'{idx}']}\nB. {file_content['opb'][f'{idx}']}\nC. {file_content['opc'][f'{idx}']}\nD. {file_content['opd'][f'{idx}']}\n"
  org_dict['answer_idx'] = file_content['answer_idx'][f'{idx}']
  org_dict['answer'] = file_content['answer'][f'{idx}']
  org_dict['explanation'] = file_content['explanation'][f'{idx}']
  org_dict['link'] = file_content['link'][f'{idx}']
  # break
  with open('./data/medbullets_op5/dev.jsonl', 'a+') as f:
    f.write(json.dumps(org_dict) + '\n')
  