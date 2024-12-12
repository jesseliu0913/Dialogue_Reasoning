import os
import json
import torch
import openai
import transformers
from datasets import load_dataset
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", torch_dtype=torch.float16, device_map="auto").to("cuda")

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

def call_hf(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=1000, num_beams=5, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if response.startswith(prompt):
        response = response[len(prompt):].strip()
    return response

f_write = open("dialogue_medqa_llama.jsonl", "a")
for idx, line in enumerate(ds):
  # if idx > 10:
  line_dict = {}
  input = f"{line['question']} {line['answer']}"
  prefix = f"""Please convert the following paragraph into a doctor-patient dialogue. Ensure that all the information provided, including personal details, symptoms, examination findings, diagnosis, and treatment, is included. Most important is the final answer, "{line['answer']}", which must be included in the dialogue without any changes. Use natural conversational language to connect the details, but do not introduce any new information. The dialogue should not be too redundant:"""
  prompt = f"{prefix}\n{input}"
  # response = call_gpt("gpt-4o", prompt)
  # response = response.replace("**", "")
  response = call_hf(prompt)
  line_dict['index'] = idx
  line_dict['answer'] = line['answer']
  line_dict['input'] = input
  line_dict['response'] = response
  f_write.write(json.dumps(line_dict) + "\n")
     