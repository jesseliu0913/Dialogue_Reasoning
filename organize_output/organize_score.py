import os
import csv
import json
import shutil


INPUT_PTAH = "/playpen/jesse/Dialogue_Reasoning/pipeline/eval_results_re/all_results.json"
input_data = json.load(open(INPUT_PTAH, 'r'))


def check_loraname(others_info, file):
    experiments_type = ''
    tuning_type = ''
    if 'exp1' in others_info:
        experiments_type = 'Exp1'
    elif 'exp2' in others_info or 'context' in others_info:
        experiments_type = 'Exp2'
    elif 'expcomb' in others_info or 'exp3' in others_info or 'baseline' in others_info:
        experiments_type = 'Exp3'
    elif 'None' in others_info:
        experiments_type = 'Raw'
    else:
        print(file)
        print("Error: Unknown experiment type", others_info)

    if 'mc' in others_info:
        tuning_type = 'Multi_Choice'
    elif 'context' in others_info:
        tuning_type = 'Case Report'
    elif 'dialogue' in others_info:
        tuning_type = 'Dialogue'
    elif 'baseline' in others_info:
        tuning_type = 'Baseline'
    elif 'None' in others_info:
        tuning_type = 'Raw'
    else:
        print(file)
        print("Error: Unknown tuning type", others_info)
    
    return experiments_type, tuning_type

rows = []
for round_type, model_info in input_data.items():
    if round_type != 'one_round':
        for model_key, cl_dict in model_info.items():
            for cl_name, file_dict in cl_dict.items():
                cl = cl_name.split(".")[0]
                for file_name, metrics in file_dict.items():
                    parts = file_name.split("_")[:-2]
                    model_type = parts[1]
                    others_info = parts[2:]
                    exp_type, tune_type = check_loraname(others_info, file_name)
                    rows.append({
                        "model":         model_type,
                        "CL":            cl,
                        "experiment":    exp_type,
                        "tuning":        tune_type,
                        "single_acc":    round(float(metrics['Signle']), 4),
                        "multi_acc":     round(float(metrics['POS']), 4)
                    })
            
with open("results_multi.csv", "w", newline="") as fout:
    writer = csv.DictWriter(fout, fieldnames=["model","CL","experiment","tuning","single_acc","multi_acc"])
    writer.writeheader()
    writer.writerows(rows)


# total_dict = {}
# for round_type in list(input_data.keys()):
#     if round_type == 'one_round':
#         model_info = input_data[round_type]
#         for model_type in list(model_info.keys()):
#             for cl_name in list(model_info[model_type].keys()):
#                 cl = cl_name.split(".")[0]
#                 for file_name in list(model_info[model_type][cl_name].keys()):
#                     file_list = file_name.split("_")[:-2]
#                     model_type = file_list[1]
#                     others_info = file_list[2:]
#                     experiments_type, tuning_type = check_loraname(others_info, file_name)

#                     print(model_info[model_type][cl_name][file_name]['Signle'])
#                     print(model_info[model_type][cl_name][file_name]['POS'])

#                     singel_acc = round(float(model_info[model_type][cl_name][file_name]['Signle']), 4)
#                     multi_acc = round(float(model_info[model_type][cl_name][file_name]['POS']), 4)

#                     print(model_type, cl, experiments_type, tuning_type)
#                     print(singel_acc, multi_acc)

#                     break
#                 break
#             break
#         break
#     else:
#         break