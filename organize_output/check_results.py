import os
import json
import shutil


RESULTS_DIR = "/playpen/jesse/Dialogue_Reasoning/organize_output/other_cluster/all/multi_round"
OUTPUT_FOLDER = "/playpen/jesse/Dialogue_Reasoning/organize_output/results/all/multi_round"
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

existing_files = []
for root, dirs, files in os.walk(OUTPUT_FOLDER):
    for file in files:
        existing_files.append(file)

for root, dirs, files in os.walk(RESULTS_DIR):
    for file in files:
        if file in existing_files:
            print(f"File {file} already exists in {OUTPUT_FOLDER}. Skipping.")
            continue
        else:
            file_path = os.path.join(root, file)
            file_list = file.split("_")[:-2]
            cl = file.split("_")[-1].replace('.json', '')
            model_type = file_list[1]
            others_info = file_list[2:]
            experiments_type, tuning_type = check_loraname(others_info, file)

            data = []
            count = 0
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    count += 1
                    # try:
                    #     data.append(json.loads(line))
                    # except json.JSONDecodeError as e:
                    #     print(f"Skipping line {line_num} due to JSONDecodeError: {e}")
            
            if count < 3356:
                print("Error: Less than 3356 lines")
                print(count)
                print(file)
            else:
                os.makedirs(OUTPUT_FOLDER, exist_ok=True)
                dest_path = os.path.join(OUTPUT_FOLDER, file)
                shutil.move(file_path, dest_path)
                # print(f"Moved: {file_path} -> {dest_path}")
            

