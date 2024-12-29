import os
import re
import ast
import json
import argparse
import numpy as np
from utils import *

parser = argparse.ArgumentParser(description="Eval the results")
parser.add_argument('--task', type=str, required=True, help='task type')
args = parser.parse_args()

if args.task == "one_round":
    FOLDER_PATH, task = "./output/llama8b_ep3/one_round", "one_round"
elif args.task == "multi_round":
    FOLDER_PATH, task= "./output/llama8b_ep3/multi_round", "multi_round"
else:
    print("Plz pass the correct task type in [one_round, multi_round]")

input_files = [f for f in os.listdir(FOLDER_PATH) if not f.startswith(".")]


def calculate_mrr(truth, predicted):
    truth = list(truth)
    predicted = list(predicted)

    truth_rank = {item: rank for rank, item in enumerate(truth)}
    
    reciprocal_ranks = []
    for rank, item in enumerate(predicted):
        if item in truth_rank:
            reciprocal_ranks.append(1 / (rank + 1))
  
    return sum(reciprocal_ranks) / len(truth) if reciprocal_ranks else 0.0


def calculate_correct_positions(truth, predicted):
    truth = list(truth)
    predicted = list(predicted)
    length = len(truth)

    correct_count = 0

    for index, item in enumerate(predicted):
        if index < len(truth) and truth[index] == item:  
            correct_count += 1

    return correct_count / length if length != 0 else 0.0


for input_f in input_files:
    if "Mistral-7B-Instruct-v0.3" not in input_f:
        print(input_f)
        precision_scores = []
        recall_scores = []
        f1_scores = []
        mrr_scores = []
        pos_socres = []

        file_path = os.path.join(FOLDER_PATH, input_f)
        file_lst = open(file_path, 'r')
        if task == "one_round":
            for idx, line in enumerate(file_lst):
                try:
                    input_data = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Skipping line {idx} due to JSONDecodeError: {e}")
                ground_truth = set(input_data['truth_idx'])
                trouble_maker = set(input_data['trouble_idx'])
                total_list = ground_truth | trouble_maker
                if find_integer(input_data['response']) is not None and len(ground_truth) != 0:
                    numbers = re.findall(r'\d+', input_data['response'])
                    output = set(list(map(int, numbers))[:len(ground_truth)])

                    # mrr_score = calculate_mrr(ground_truth, output)
                    pos_socre = calculate_correct_positions(ground_truth, output)
                    # calculate the confusion matrix
                    tp = ground_truth & output
                    fp = output - ground_truth
                    fn = ground_truth - output
                    tn = total_list - (ground_truth | output)

                    precision = len(tp) / (len(tp) + len(fp)) if (len(tp) + len(fp)) > 0 else 0
                    recall = len(tp) / (len(tp) + len(fn)) if (len(tp) + len(fn)) > 0 else 0
                    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                    precision_scores.append(precision)
                    recall_scores.append(recall)
                    f1_scores.append(f1_score)
                    # mrr_scores.append(mrr_score)
                    pos_socres.append(pos_socre)

            print("F1:", np.mean(np.array(f1_scores)))
            print("Precision:", np.mean(np.array(precision_scores)))
            print("Recall:", np.mean(np.array(recall_scores)))
            # print("MRR:", np.mean(np.array(mrr_scores)))
            print("POS:", np.mean(np.array(pos_socres)))

        elif task == "multi_round":
            for idx, line in enumerate(file_lst):
                input_data = json.loads(line)
                ground_truth = set(input_data['truth_idx'])
                trouble_maker = set(input_data['trouble_idx'])
                total_list = ground_truth | trouble_maker
                if len(ground_truth) != 0:
                    output = input_data['response_index']
                    if "$" in output: 
                        output.remove("$")
                    output = set(output)
                    # mrr_score = calculate_mrr(ground_truth, output)
                    pos_socre = calculate_correct_positions(ground_truth, output)

                    # calculate the confusion matrix
                    tp = ground_truth & output
                    fp = output - ground_truth
                    fn = ground_truth - output
                    tn = total_list - (ground_truth | output)

                    precision = len(tp) / (len(tp) + len(fp)) if (len(tp) + len(fp)) > 0 else 0
                    recall = len(tp) / (len(tp) + len(fn)) if (len(tp) + len(fn)) > 0 else 0
                    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                    precision_scores.append(precision)
                    recall_scores.append(recall)
                    f1_scores.append(f1_score)
                    # mrr_scores.append(mrr_score)
                    pos_socres.append(pos_socre)

            print("F1:", np.mean(np.array(f1_scores)))
            print("Precision:", np.mean(np.array(precision_scores)))
            print("Recall:", np.mean(np.array(recall_scores)))
            # print("MRR:", np.mean(np.array(mrr_scores)))
            print("POS:", np.mean(np.array(pos_socres)))