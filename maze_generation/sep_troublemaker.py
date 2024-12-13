import os,re
import json


def extract_info(data):
    pattern = r'"([^"]+)"\s*:\s*\[([^\]]*)\]'
    matches = re.findall(pattern, data)
    info_dict = {key: [int(x) for x in value.split(',') if x.strip().isdigit()] for key, value in matches}
    return info_dict

file_path = "./output/raw_response.jsonl"
f_read = open(file_path, 'r')
f_write = open("./output/dealed_response.jsonl", 'w')
for line in f_read:
    line_dict = {}
    # line_dict = {"Patient Personal Background": [], "Patient Symptoms": [], "Examination and Results": [], "Treatment": [1], "Diagnosis": [],"Others": [0, 2, 3]}
    data = json.loads(line.strip())
    response = extract_info(data['response'])
    zoo = data['zoo']
    for res_key in response:
        sent_idx = response[res_key]
        if sent_idx != []: 
            sent_lst = [zoo[int(idx)] for idx in sent_idx if int(idx) < len(zoo)]
            line_dict[res_key] = sent_lst
        
    json.dump(line_dict, f_write)
    f_write.write("\n")
