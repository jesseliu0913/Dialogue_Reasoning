import os
import openai

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from peft import PeftModel, PeftConfig


def init_model(model_name: str, model_args: str, task_name: str, lora_weight=None):
    model_weight = model_args
    # load tokenizer
    if task_name == "medalpaca":
      tokenizer = LlamaTokenizer.from_pretrained(model_weight)
    elif task_name == "chatgpt":
      tokenizer = None
    elif lora_weight != None:
      tokenizer = AutoTokenizer.from_pretrained(lora_weight)
    else:
      tokenizer = AutoTokenizer.from_pretrained(model_weight)

    # load the model
    if model_name ==  "hf":
        if lora_weight != None:
            model = AutoModelForCausalLM.from_pretrained(lora_weight, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True, trust_remote_code=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_weight, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True, trust_remote_code=True)
       

    elif model_name == "openai":
      api_key = os.getenv("OPENAI_API_KEY")
      if api_key is None:
          raise ValueError("OpenAI API key not found. Please set the 'OPENAI_API_KEY' environment variable.")
      else:
          model = "openai"
          tokenizer = None
    else:
        print("Input the model name")
        model, tokenizer = None, None
    
    return model, tokenizer

def call_gpt(tokenizer: str, model_args: str, message: str, temperature=0.7, max_new_tokens=100, top_p=0.9):
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

def call_model(tokenizer, model, init_prompt, max_new_tokens, openai=True):
    if openai:
        return call_gpt(tokenizer, model, init_prompt, max_new_tokens=max_new_tokens)
    else:
        inputs = tokenizer(init_prompt, return_tensors="pt").to('cuda')
        input_length = inputs['input_ids'].shape[1]
        with torch.no_grad():
            response = model.generate(**inputs, max_new_tokens=50)
        return tokenizer.decode(response[0][input_length:], skip_special_tokens=True)
