import os
import re
import json
import numpy as np
from collections import defaultdict
from utils import *

task_types = ["basic", "advance", "challenge"]
round_types = ["one_round", "multi_round"]

model_prefixes = [
    "Llama-3.2-3B-Instruct",
    "Llama-3.1-8B-Instruct",
    "Qwen2.5-3B-Instruct",
    "Mistral-7B-Instruct-v0.3"
]

def calculate_mrr(truth, predicted):
    truth = list(truth)
    predicted = list(predicted)
    truth_rank = {item: rank for rank, item in enumerate(truth)}
    reciprocal_ranks = [1 / (rank + 1) for rank, item in enumerate(predicted) if item in truth_rank]
    return sum(reciprocal_ranks) / len(truth) if reciprocal_ranks else 0.0

def calculate_correct_positions(truth, predicted):
    truth = list(truth)
    predicted = list(predicted)
    length = len(truth)
    correct_count = sum(1 for index, item in enumerate(predicted[:length]) if truth[index] == item)
    return correct_count / length if length != 0 else 0.0

for task in task_types:
    FOLDER_PATH = f"./output/{task}"
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
                            "F1": [], "Precision": [], "Recall": [], "POS": []
                        }

                    precision_scores = []
                    recall_scores = []
                    f1_scores = []
                    pos_scores = []

                    file_path = os.path.join(round_folder, input_f)
                    with open(file_path, 'r') as file_lst:
                        for idx, line in enumerate(file_lst):
                            try:
                                input_data = json.loads(line)
                            except json.JSONDecodeError as e:
                                print(f"Skipping line {idx} due to JSONDecodeError: {e}")
                                continue

                            ground_truth = set(input_data['truth_idx'])
                            trouble_maker = set(input_data['trouble_idx'])
                            total_list = ground_truth | trouble_maker

                            if round_type == "one_round":
                                if find_integer(input_data['response']) is not None and len(ground_truth) != 0:
                                    numbers = re.findall(r'\d+', input_data['response'])
                                    output = set(map(int, numbers[:len(ground_truth)]))

                                    pos_score = calculate_correct_positions(ground_truth, output)
                                    tp = ground_truth & output
                                    fp = output - ground_truth
                                    fn = ground_truth - output
                                    precision = len(tp) / (len(tp) + len(fp)) if (len(tp) + len(fp)) > 0 else 0
                                    recall = len(tp) / (len(tp) + len(fn)) if (len(tp) + len(fn)) > 0 else 0
                                    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                                    precision_scores.append(precision)
                                    recall_scores.append(recall)
                                    f1_scores.append(f1_score)
                                    pos_scores.append(pos_score)

                            elif round_type == "multi_round":
                                if len(ground_truth) != 0:
                                    output = set(input_data['response_index'])
                                    if "$" in output:
                                        output.remove("$")

                                    pos_score = calculate_correct_positions(ground_truth, output)
                                    tp = ground_truth & output
                                    fp = output - ground_truth
                                    fn = ground_truth - output
                                    precision = len(tp) / (len(tp) + len(fp)) if (len(tp) + len(fp)) > 0 else 0
                                    recall = len(tp) / (len(tp) + len(fn)) if (len(tp) + len(fn)) > 0 else 0
                                    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                                    precision_scores.append(precision)
                                    recall_scores.append(recall)
                                    f1_scores.append(f1_score)
                                    pos_scores.append(pos_score)

                    results[round_type][prefix][confusion_level][input_f]["F1"].append(np.mean(f1_scores) if f1_scores else 0.0)
                    results[round_type][prefix][confusion_level][input_f]["Precision"].append(np.mean(precision_scores) if precision_scores else 0.0)
                    results[round_type][prefix][confusion_level][input_f]["Recall"].append(np.mean(recall_scores) if recall_scores else 0.0)
                    results[round_type][prefix][confusion_level][input_f]["POS"].append(np.mean(pos_scores) if pos_scores else 0.0)

    final_results = {
        "one_round": {
            prefix: {
                confusion_level: {
                    file_name: {
                        "F1": np.mean(metrics["F1"]) if metrics["F1"] else 0.0,
                        "Precision": np.mean(metrics["Precision"]) if metrics["Precision"] else 0.0,
                        "Recall": np.mean(metrics["Recall"]) if metrics["Recall"] else 0.0,
                        "POS": np.mean(metrics["POS"]) if metrics["POS"] else 0.0
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
                        "F1": np.mean(metrics["F1"]) if metrics["F1"] else 0.0,
                        "Precision": np.mean(metrics["Precision"]) if metrics["Precision"] else 0.0,
                        "Recall": np.mean(metrics["Recall"]) if metrics["Recall"] else 0.0,
                        "POS": np.mean(metrics["POS"]) if metrics["POS"] else 0.0
                    }
                    for file_name, metrics in files.items()
                }
                for confusion_level, files in results["multi_round"][prefix].items()
            }
            for prefix in model_prefixes
        }
    }

    output_dir = "./eval_results"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{task}_results.json")

    with open(output_file, 'w') as json_file:
        json.dump(final_results, json_file, indent=4)

    print(f"Evaluation results for {task} saved to {output_file}")
