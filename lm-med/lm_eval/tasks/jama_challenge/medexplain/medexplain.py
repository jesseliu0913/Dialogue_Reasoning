import collections
import re
import string
from rouge_score import rouge_scorer
from bert_score import score as bert_score


def doc_to_text(doc) -> str:
    propmt = f"""
Q: You are a medical expert that just answered
the above question. Please explain why {doc["answer_idx"]}
is correct while the rest choices are incorrect. You
should explain each choice in detail."""

    option_choices = {
        "A": doc["opa"],
        "B": doc["opb"],
        "C": doc["opc"],
        "D": doc["opd"],
    }
    answers = "".join((f"{k}. {v}\n") for k, v in option_choices.items())
    input = f"""
    QUESTION: {doc['question']}
    ANSWER CHOICES: {answers}
    ANSWER: {doc['answer_idx']}
    Q: {propmt}
    A: """
    return input

def doc_to_target(doc) -> int:
    return doc['explanation']

def normalize_answer(s):
    """Lower text and remove punctuation, articles, and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the|un|une|des|le|la|les)\b", re.UNICODE)
        return re.sub(regex, " ", text)
    
    def white_space_fix(text):
        return " ".join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()

def flatten_list(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))  
        else:
            flat_list.append(item)
    return flat_list

def compute_rouge_l(prediction, ground_truth):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(prediction, ground_truth)
    return scores['rougeL'].fmeasure

def compute_bert_score(prediction, ground_truth):
    P, R, F1 = bert_score([prediction], [ground_truth], lang="en")
    return F1.mean().item()


def compute_f1(prediction, ground_truth):
    pred_tokens = get_tokens(prediction)
    gt_tokens = get_tokens(ground_truth)
    common = collections.Counter(pred_tokens) & collections.Counter(gt_tokens)
    num_same = sum(common.values())
    
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return int(pred_tokens == gt_tokens)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def compute_bleu(prediction, ground_truth):
    pred_tokens = get_tokens(prediction)
    gt_tokens = get_tokens(ground_truth)
    overlap = collections.Counter(pred_tokens) & collections.Counter(gt_tokens)
    overlap_count = sum(overlap.values())
    if len(pred_tokens) == 0:
        return 0.0
    precision = overlap_count / len(pred_tokens)
    bp = min(1.0, len(pred_tokens) / len(gt_tokens)) if len(gt_tokens) > 0 else 1.0
    bleu = bp * precision
    return bleu

def process_results(doc, results) -> str:
    results = " ".join(flatten_list(results)) 
    groundtruth = str(doc['explanation'])

    return {
        "f1": compute_f1(results, groundtruth),
        "bleu": compute_bleu(results, groundtruth),
        "rouge_l": compute_rouge_l(results, groundtruth),
        "bert_score": compute_bert_score(results, groundtruth),
    }
