def doc_to_text(doc) -> str:
    option_choices = {
        "A": doc["opa"],
        "B": doc["opb"],
        "C": doc["opc"],
        "D": doc["opd"],
    }
    answers = "".join((f"{k}. {v}\n") for k, v in option_choices.items())
    return f"Question: {doc['question']}\n{answers}Answer:"


def doc_to_target(doc) -> int:
    return doc["answer_idx"]