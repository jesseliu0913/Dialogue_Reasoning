import os
import json
import time
import openai
import random
import datasets
from datasets import load_dataset


def call_gpt(model_args: str, message: str, temperature=0.7, max_new_tokens=1000, top_p=0.9):
    openai.api_key = os.getenv("OPENAI_API_KEY")

    for attempt in range(3):  
      try:  
        response = openai.ChatCompletion.create(
          model=model_args,
          messages=[
              {"role": "system", "content": "You are a helpful assistant in medical task."},
              {"role": "user", "content": f"{message}"}
          ],
          temperature=temperature,
          max_tokens=max_new_tokens,
          top_p=top_p,
        )
        return response['choices'][0]['message']['content']
        
      except openai.error.Timeout as e:
            print(f"Timeout occurred on attempt {attempt + 1}: {e}")
            time.sleep(5)  
        
      except openai.error.OpenAIError as e:
            print(f"An OpenAI API error occurred: {e}")
            break

prefix = """
Provide only the indices corresponding to the following categories based on the listed statements:

Patient Personal Background: Include information about the patient’s lifestyle, habits, past medical history, pregnancy history, or ongoing medications. Exclude any details related to current symptoms or findings.
Patient Symptoms: Include indices for any current complaints, observed signs, or reported symptoms (e.g., pain, contractions, fluid discharge). Focus on present or active conditions.
Examination and Results: Include indices for findings from physical examinations, test results, or clinical measurements (e.g., vital signs, imaging, or laboratory results).
Treatment: Include indices for any treatments, medications, or procedures prescribed or initiated during this clinical encounter.
Diagnosis: Include indices only if the text explicitly states or strongly implies a diagnosis.
Others: Include indices for any information that does not fit the above categories (e.g., general statements or irrelevant details).
Do not include the full sentences, only the indices corresponding to each category.

## Output Format

Provide your evaluation as a JSON object with the following structure: \n
json\n{\n    "Patient Personal Background":  [],\n Patient Symptoms: [],\n Examination and Results: [],\n  Treatment: [],\n Diagnosis: [],\n Others: [],\n }
"""

dataset = "JesseLiu/MedQA_Maze"
ds = load_dataset(dataset)
f_write = open("./output/raw_response.jsonl", "a+")
for idx, item in enumerate(ds['test']):
  if idx >= 479 :
    raw_response = {}
    prompt = f"{item['groundtruth_zoo']}\n{prefix}\n"
    response = call_gpt("gpt-4o", prompt)
    
    raw_response['prerequisit'] = item['prerequisit']
    raw_response['zoo'] = item['groundtruth_zoo']
    raw_response['response'] = response

    f_write.write(json.dumps(raw_response) + '\n')




