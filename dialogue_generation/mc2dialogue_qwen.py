import os
import json
import torch
import openai
import transformers
from datasets import load_dataset
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", padding_side='left')
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"  
)

ds = load_dataset("GBaker/MedQA-USMLE-4-options")['train']
ds = ds.select(range(6777, len(ds)))

def call_hf_batch(prompts):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(
        **inputs,
        max_new_tokens=1000,
        num_beams=5,
        temperature=0.7
    )
    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    cleaned_responses = []
    for prompt, response in zip(prompts, responses):
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        cleaned_responses.append(response)
    return cleaned_responses

batch_size = 4
batch_prompts = []
batch_indices = []
batch_answers = []
batch_inputs = []

with open("dialogue_medqa_qwen.jsonl", "a") as f_write:
    for idx, line in enumerate(ds):
        idx = idx + 6777
        input_text = f"{line['question']} {line['answer']}"
        prefix = (
            f"Please convert the following paragraph into a doctor-patient dialogue. "
            f"Ensure that all the information provided, including personal details, symptoms, "
            f"examination findings, diagnosis, and treatment, is included. Most important is the "
            f"final answer, \"{line['answer']}\", which must be included in the dialogue without any changes. "
            f"Use natural conversational language to connect the details, but do not introduce any new information. "
            f"The dialogue should not be too redundant:"
        )
        prompt = f"{prefix}\n{input_text}"

        batch_prompts.append(prompt)
        batch_indices.append(idx)
        batch_answers.append(line['answer'])
        batch_inputs.append(input_text)

        if len(batch_prompts) == batch_size:
            responses = call_hf_batch(batch_prompts)
            for i in range(len(responses)):
                line_dict = {
                    'index': batch_indices[i],
                    'answer': batch_answers[i],
                    'input': batch_inputs[i],
                    'response': responses[i]
                }
                f_write.write(json.dumps(line_dict) + "\n")
            batch_prompts = []
            batch_indices = []
            batch_answers = []
            batch_inputs = []

    if batch_prompts:
        responses = call_hf_batch(batch_prompts)
        for i in range(len(responses)):
            line_dict = {
                'index': batch_indices[i],
                'answer': batch_answers[i],
                'input': batch_inputs[i],
                'response': responses[i]
            }
            f_write.write(json.dumps(line_dict) + "\n")
