from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

lora_weights = "JesseLiu/llama_dialogue_ep4"
model = PeftModel.from_pretrained(model, lora_weights)

input_text = "Hello, how can I assist you?"
inputs = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


