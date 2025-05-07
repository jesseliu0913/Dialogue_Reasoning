import os
import re
import json
import numpy as np
from collections import defaultdict
from utils import *

# task_types = ["basic", "advance", "challenge"]
task_types = ['all']
round_types = ["one_round", "multi_round"]

model_prefixes = [
    "Llama-3.2-3B-Instruct",
    "Llama-3.1-8B-Instruct",
    "Qwen2.5-3B-Instruct",
]

def calculate_mrr(truth, predicted):
    truth = list(truth)
    predicted = list(predicted)
    truth_rank = {item: rank for rank, item in enumerate(truth)}
    reciprocal_ranks = [1 / (rank + 1) for rank, item in enumerate(predicted) if item in truth_rank]
    return sum(reciprocal_ranks) / len(truth) if reciprocal_ranks else 0.0

def calculate_pairwise_accuracy(truth, predicted):
    truth = list(truth)
    predicted = list(predicted)
    truth_pairs = [(truth[i], truth[i+1]) for i in range(len(truth) - 1)]
    predicted_pairs = [
        (predicted[i], predicted[i+1]) for i in range(len(predicted) - 1)
    ] + [
        (predicted[i+1], predicted[i]) for i in range(len(predicted) - 1)
    ]
    
    match_count = sum(1 for pair in predicted_pairs if pair in truth_pairs)
    
    return match_count / len(truth_pairs) if truth_pairs else 0.0


def calculate_correct_positions(truth, predicted):
    truth = list(truth)
    predicted = list(predicted)
    length = len(truth)

    correct_count = 0

    for index, item in enumerate(predicted):
        if index < len(truth) and truth[index] == item:  
            correct_count += 1

    return correct_count / length if length != 0 else 0.0

for task in task_types:
    FOLDER_PATH = f"/playpen/jesse/Dialogue_Reasoning/organize_output/results/{task}"
    results = {
        "one_round": {prefix: defaultdict(dict) for prefix in model_prefixes},
        "multi_round": {prefix: defaultdict(dict) for prefix in model_prefixes}
    }

    for round_type in round_types:
        round_folder = os.path.join(FOLDER_PATH, round_type)
        if not os.path.exists(round_folder):
            print(f"Folder {round_folder} does not exist, skipping...")
            continue

        input_files = [f for f in os.listdir(round_folder) if not f.startswith(".")]

        for input_f in input_files:
            for prefix in model_prefixes:
                if prefix in input_f:

                    confusion_level = input_f.split('_')[-1]

                    if input_f not in results[round_type][prefix][confusion_level]:
                        results[round_type][prefix][confusion_level][input_f] = {
                            "POS": [], "Single": []
                        }

                    precision_scores = []
                    recall_scores = []
                    f1_scores = []
                    pos_scores = []
                    single_scores = []

                    file_path = os.path.join(round_folder, input_f)
                    with open(file_path, 'r') as file_lst:
                        for idx, line in enumerate(file_lst):
                            try:
                                input_data = json.loads(line)
                            except json.JSONDecodeError as e:
                                print(f"Skipping line {idx} due to JSONDecodeError: {e}")
                                continue

                            ground_truth = input_data['truth_idx']
                            trouble_maker = input_data['trouble_idx']

                            if round_type == "one_round":
                                if len(ground_truth) != 0:
                                    if find_integer(input_data['response']) is not None:
                                        numbers = re.findall(r'\d+', input_data['response'])
                                        output = list(dict.fromkeys(map(int, numbers)))[:len(ground_truth)]

                                        pos_score = calculate_correct_positions(ground_truth, output)
                                        single_score = calculate_pairwise_accuracy(ground_truth, output)
                                        pos_scores.append(pos_score)
                                        single_scores.append(single_score)
                                    else:
                                        pos_scores.append(0)
                                        single_scores.append(0)

                            elif round_type == "multi_round":
                                if len(ground_truth) != 0:
                                    output = input_data['response_index']
                                    if "$" in output:
                                        output.remove("$")

                                    pos_score = calculate_correct_positions(ground_truth, output)
                                    single_score = calculate_pairwise_accuracy(ground_truth, output)

                                    pos_scores.append(pos_score)
                                    single_scores.append(single_score)

                    # results[round_type][prefix][confusion_level][input_f]["F1"].append(np.mean(f1_scores) if f1_scores else 0.0)
                    # results[round_type][prefix][confusion_level][input_f]["Precision"].append(np.mean(precision_scores) if precision_scores else 0.0)
                    # results[round_type][prefix][confusion_level][input_f]["Recall"].append(np.mean(recall_scores) if recall_scores else 0.0)
                    results[round_type][prefix][confusion_level][input_f]["POS"].append(np.mean(pos_scores) if pos_scores else 0.0)
                    results[round_type][prefix][confusion_level][input_f]["Single"].append(np.mean(single_scores) if single_scores else 0.0)

    final_results = {
        "one_round": {
            prefix: {
                confusion_level: {
                    file_name: {
                        # "F1": np.mean(metrics["F1"]) if metrics["F1"] else 0.0,
                        # "Precision": np.mean(metrics["Precision"]) if metrics["Precision"] else 0.0,
                        # "Recall": np.mean(metrics["Recall"]) if metrics["Recall"] else 0.0,
                        "POS": np.mean(metrics["POS"]) if metrics["POS"] else 0.0,
                        "Signle": np.mean(metrics["Single"]) if metrics["Single"] else 0.0
                    }
                    for file_name, metrics in files.items()
                }
                for confusion_level, files in results["one_round"][prefix].items()
            }
            for prefix in model_prefixes
        },
        "multi_round": {
            prefix: {
                confusion_level: {
                    file_name: {
                        # "F1": np.mean(metrics["F1"]) if metrics["F1"] else 0.0,
                        # "Precision": np.mean(metrics["Precision"]) if metrics["Precision"] else 0.0,
                        # "Recall": np.mean(metrics["Recall"]) if metrics["Recall"] else 0.0,
                        "POS": np.mean(metrics["POS"]) if metrics["POS"] else 0.0,
                        "Signle": np.mean(metrics["Single"]) if metrics["Single"] else 0.0
                    }
                    for file_name, metrics in files.items()
                }
                for confusion_level, files in results["multi_round"][prefix].items()
            }
            for prefix in model_prefixes
        }
    }

    output_dir = "./eval_results_re"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{task}_results.json")

    with open(output_file, 'w') as json_file:
        json.dump(final_results, json_file, indent=4)

    print(f"Evaluation results for {task} saved to {output_file}")
