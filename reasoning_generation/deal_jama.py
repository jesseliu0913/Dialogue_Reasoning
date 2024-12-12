import json


with open('/scratch0/zx22/zijie/MDAgents/data/jama_raw.json', 'r') as file:
    jama_content = json.load(file)

for item in jama_content[10: ]:
    org_dict = {}
    org_dict['question'] = item['question']
    org_dict['opa'] = item['opa']
    org_dict['opb'] = item['opb']
    org_dict['opc'] = item['opc']
    org_dict['opd'] = item['opd']

    org_dict['answer_idx'] = item['answer_idx']
    org_dict['answer'] = item['answer']
    org_dict['explanation'] = item['explanation']
    
    with open('./data/jama_challenge/test.jsonl', 'a+') as f:
      f.write(json.dumps(org_dict) + '\n')
  