def doc_to_text(doc) -> str:
    option_choices = {
        "A": doc["choicesA"],
        "B": doc["choicesB"],
        "C": doc["choicesC"],
        "D": doc["choicesD"],
        "E": doc["choicesE"],
    }
    answers = "".join((f"{k}. {v}\n") for k, v in option_choices.items())
    return f"Question: {doc['question']}\n{answers}Answer:"


def doc_to_target(doc) -> int:
    return doc["answer_idx"]