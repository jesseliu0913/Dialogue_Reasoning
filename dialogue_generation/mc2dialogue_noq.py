import os
import json
import openai
from datasets import load_dataset

ds = load_dataset("GBaker/MedQA-USMLE-4-options")['train']

def call_gpt(model_args: str, message: str, temperature=0.7, max_new_tokens=1000, top_p=0.9):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    response = openai.ChatCompletion.create(
      model=model_args,
      messages=[
          {"role": "system", "content": "You are a helpful assistant in medical reasoning task."},
          {"role": "user", "content": f"{message}"}
      ],
      temperature=temperature,
      max_tokens=max_new_tokens,
      top_p=top_p,
    ) 

    return response['choices'][0]['message']['content']

f_write = open("dialogue_medqa_noq.jsonl", "a")
for idx, line in enumerate(ds):
  if idx <= 1000:
    line_dict = {}
    former_sentence = ".".join(line['question'].split(".")[0:-1])
    last_sentence = line['question'].split(".")[-1]
    prefix = f"""Please convert the following paragraph into a doctor-patient dialogue. Ensure that all the information provided, including personal details, symptoms, examination findings, diagnosis, and treatment, is included. Use natural conversational language to connect the details, but do not introduce any new information. The dialogue should not be too redundant:"""
    prompt = f"{prefix}\n{former_sentence}."
    response = call_gpt("gpt-4o", prompt)
    response = response.replace("**", "")
    line_dict['index'] = idx
    line_dict['answer'] = line['answer']
    line_dict['input'] = former_sentence
    line_dict['question'] = last_sentence
    line_dict['response'] = response
    f_write.write(json.dumps(line_dict) + "\n")
     
