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
    FOLDER_PATH, task = "./output/advance/one_round", "one_round"
elif args.task == "multi_round":
    FOLDER_PATH, task= "./output/all/multi_round", "multi_round"
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

def calculate_pairwise_accuracy(truth, predicted):
    truth = list(truth)
    predicted = list(predicted)
    # print(predicted, truth)
    truth_pairs = [(truth[i], truth[i+1]) for i in range(len(truth) - 1)]
    predicted_pairs = [
        (predicted[i], predicted[i+1]) for i in range(len(predicted) - 1)
    ] + [
        (predicted[i+1], predicted[i]) for i in range(len(predicted) - 1)
    ]
    
    match_count = sum(1 for pair in predicted_pairs if pair in truth_pairs)
    # print("match_count", match_count)
    # print("truth_pairs", truth_pairs)
    
    return match_count / len(truth_pairs) if truth_pairs else 0.0


def calculate_correct_positions(truth, predicted):
    truth = list(truth)
    predicted = list(predicted)
    length = len(truth)
    # print(predicted, truth)

    correct_count = 0

    for index, item in enumerate(predicted):
        if index < len(truth) and truth[index] == item:  
            correct_count += 1

    # print("correct_count", correct_count)
    # print("length", length)

    return correct_count / length if length != 0 else 0.0


for input_f in input_files:
    # hf_Qwen2.5-3B-Instruct_qwen25_combine_dialogue_0_3
    # hf_Qwen2.5-3B-Instruct_qwen3b_baseline_0_3
    # hf_Llama-3.2-3B-Instruct_None_0_3
    # hf_Llama-3.2-3B-Instruct_llama32_combine_dialogue_0_3
    if "hf_Qwen2.5-3B-Instruct_qwen3b_baseline_0_3" in input_f:
        print(input_f)
        precision_scores = []
        recall_scores = []
        f1_scores = []
        mrr_scores = []
        pos_socres = []
        single_scores = []

        file_path = os.path.join(FOLDER_PATH, input_f)
        file_lst = open(file_path, 'r')
        if task == "one_round":
            for idx, line in enumerate(file_lst):
                try:
                    input_data = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Skipping line {idx} due to JSONDecodeError: {e}")
                ground_truth = input_data['truth_idx']
                trouble_maker = input_data['trouble_idx']
                # total_list = ground_truth - trouble_maker
                if len(ground_truth) != 0:
                    if find_integer(input_data['response']) is not None:
                        numbers = re.findall(r'\d+', input_data['response'])
                        output = list(dict.fromkeys(map(int, numbers)))[:len(ground_truth)]

                        # mrr_score = calculate_mrr(ground_truth, output)
                        pos_socre = calculate_correct_positions(ground_truth, output)
                        single_score = calculate_pairwise_accuracy(ground_truth, output)
                        # calculate the confusion matrix
                        # mrr_scores.append(mrr_score)
                        pos_socres.append(pos_socre)
                        single_scores.append(single_score)
                    else:
                        pos_socres.append(0)
                        single_scores.append(0)

            # print("F1:", np.mean(np.array(f1_scores)))
            # print("Precision:", np.mean(np.array(precision_scores)))
            # print("Recall:", np.mean(np.array(recall_scores)))
            # print("MRR:", np.mean(np.array(mrr_scores)))
            print("POS:", np.mean(np.array(pos_socres)))
            print("SINGLE:", np.mean(np.array(single_scores)))

        elif task == "multi_round":
            for idx, line in enumerate(file_lst):
                input_data = json.loads(line)
                ground_truth = input_data['truth_idx']
                trouble_maker = input_data['trouble_idx']
                # total_list = ground_truth - trouble_maker
                if len(ground_truth) != 0:
                    output = input_data['response_index']
                    if "$" in output: 
                        output.remove("$")
                    # mrr_score = calculate_mrr(ground_truth, output)
                    pos_socre = calculate_correct_positions(ground_truth, output)
                    single_score = calculate_pairwise_accuracy(ground_truth, output)

                    # mrr_scores.append(mrr_score)
                    pos_socres.append(pos_socre)
                    single_scores.append(single_score)

            # print("F1:", np.mean(np.array(f1_scores)))
            # print("Precision:", np.mean(np.array(precision_scores)))
            # print("Recall:", np.mean(np.array(recall_scores)))
            # # print("MRR:", np.mean(np.array(mrr_scores)))
            print("POS:", np.mean(np.array(pos_socres)))
            print("SINGLE:", np.mean(np.array(single_scores)))
