def doc_to_text(doc) -> str:
    option_choices = {
        "A": doc["choicesA"],
        "B": doc["choicesB"],
        "C": doc["choicesC"],
        "D": doc["choicesD"],
        "E": doc["choicesE"],
    }
    answers = "".join((f"{k}. {v}\n") for k, v in option_choices.items())
    input = f"""
QUESTION: {doc['question']}
ANSWER CHOICES: {answers}
Let's think step by step and walk through all the choices in detail {doc['explanation']}
Therefore, among (A) through (E), the answer is
"""
    return input


def doc_to_target(doc) -> int:
    return doc["answer_idx"]