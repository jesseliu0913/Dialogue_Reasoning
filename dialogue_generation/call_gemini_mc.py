import os
import json
import torch
import google.generativeai as genai
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

ds = load_dataset("GBaker/MedQA-USMLE-4-options")['train']
genai.configure(api_key="AIzaSyDiLNbjetIFemfJCilKS9gboVZH1PSGVjU")
# AIzaSyDHwCBvUG0GYF6S1LNiv4LC-1bZT-UFauI
# AIzaSyB2GIsp9o0emOw3DBDqkWG29Dug4u978gc
# AIzaSyAbogSNYhQP1HXIgXBBGIpMQvfdfOAAc1I

def call_gemini(message, temperature=0.7, max_output_tokens=1000, top_p=0.9, max_retries=10, initial_delay=2):
    import time
    import random
    from google.api_core import exceptions
    
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    for attempt in range(max_retries + 1):
        try:
            response = model.generate_content(
                contents=[
                    {
                        "role": "user",
                        "parts": [{"text": message}]
                    }
                ],
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens,
                    "top_p": top_p
                }
            )
            
            return response.text
            
        except (exceptions.ResourceExhausted, exceptions.ServiceUnavailable, 
                exceptions.TooManyRequests, exceptions.DeadlineExceeded) as e:
            if attempt == max_retries:
                raise
                
            delay = initial_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"Rate limit hit. Retrying in {delay:.2f} seconds... (Attempt {attempt+1}/{max_retries})")
            time.sleep(delay)
        
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise

# def call_hf(prompt):
#     """
#     Call the Hugging Face model with the provided prompt.
#     """
#     inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
#     outputs = model.generate(**inputs, max_new_tokens=1000, num_beams=5, temperature=0.7)
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     if response.startswith(prompt):
#         response = response[len(prompt):].strip()
#     return response

existing_lines = 0
with open("dialogue_medqa_gemini.jsonl", "r") as f_read:
    for _ in f_read:
        existing_lines += 1

f_write = open("dialogue_medqa_gemini.jsonl", "a")

sliced_ds = ds.select(range(existing_lines, len(ds)))
for idx, line in enumerate(sliced_ds):
    actual_idx = idx + existing_lines
    line_dict = {}
    input_text = f"{line['question']} {line['answer']}"
    prefix = f"""Please convert the following paragraph into a doctor-patient dialogue. Ensure that all the information provided, including personal details, symptoms, examination findings, diagnosis, and treatment, is included. Most important is the final answer, "{line['answer']}", which must be included in the dialogue without any changes. Use natural conversational language to connect the details, but do not introduce any new information. The dialogue should not be too redundant:"""
    
    prompt = f"{prefix}\n{input_text}"
    response = call_gemini(prompt)
    
    line_dict['index'] = actual_idx
    line_dict['answer'] = line['answer']
    line_dict['input'] = input_text
    line_dict['response'] = response
    
    f_write.write(json.dumps(line_dict) + "\n")
f_write.close()

# nohup python call_gemini_mc.py > dilaogue.log 2>&1 &