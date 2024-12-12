def doc_to_text(doc) -> str:
    option_choices = {
        "A": doc["choicesA"],
        "B": doc["choicesB"],
        "C": doc["choicesC"],
        "D": doc["choicesD"],
    }
    answers = "".join((f"{k}. {v}\n") for k, v in option_choices.items())
    input = f"""
QUESTION: {doc['question']}
ANSWER CHOICES: {answers}
Let's think step by step and walk through all the choices in detail {doc['explanation']}
Therefore, among (A) through (D), the answer is
"""
    return input


def doc_to_target(doc) -> str:
    return doc["answer_idx"]


# def flatten_list(nested_list):
#     flat_list = []
#     for item in nested_list:
#         if isinstance(item, list):
#             flat_list.extend(flatten_list(item))  
#         else:
#             flat_list.append(item)
#     return flat_list

# def process_results(doc, results) -> str:
#     print(results)
#     return " ".join(flatten_list(results)) 
