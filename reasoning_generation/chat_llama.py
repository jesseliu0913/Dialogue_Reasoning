# Use a pipeline as a high-level helper
from transformers import pipeline

prompt = """
QUESTION: A 42-year-old woman is enrolled in a randomized controlled trial to study cardiac function in the setting of several different drugs. She is started on verapamil and instructed to exercise at 50% of her VO2 max while several cardiac parameters are being measured. During this experiment, which of the following represents the relative conduction speed through the heart from fastest to slowest?\n    ANSWER CHOICES: A. AV node > ventricles > atria > Purkinje fibers\nB. Purkinje fibers > ventricles > atria > AV node\nC. Purkinje fibers > atria > ventricles > AV node\nD. Purkinje fibers > AV node > ventricles > atria\n\n    ANSWER: C\n    Q: \nQ: You are a medical expert that just answered\nthe above question. Please explain why C\nis correct while the rest choices are incorrect. You\nshould explain each choice in detail.\n    A:
"""

# messages = [
#     {"role": "user", "content": f"{prompt}"},
# ]
# pipe = pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf", device='cuda')
# res = pipe(messages)
# print(res)

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")


inputs = tokenizer(prompt, return_tensors="pt")
output = model.generate(**inputs, max_length=2048, temperature=0.7)
output_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("Model Output:", output_text)

