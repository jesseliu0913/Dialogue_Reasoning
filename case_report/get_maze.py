import os
import re
import json
from transformers import AutoTokenizer, AutoModelForCausalLM


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
model = model.to("cuda")

INPUT_FOLDER = "./input/case_report"
FULL_FOLDER = "./input/PMC_patient_data"
OUTPUT_FILE = "./output/qualified_cases.jsonl"

input_files = [f for f in os.listdir(INPUT_FOLDER) if not f.startswith('.')]
full_files = [f for f in os.listdir(FULL_FOLDER) if not f.startswith('.')]
q_pid = []
exist_pid = []

with open(OUTPUT_FILE, "r") as f_read:
    for line in f_read:
        line = json.loads(line)
        exist_pid.append(line['pid'])

with open("./output/tuning/rare_case.jsonl", "r") as f_read:
    for line in f_read:
        line = json.loads(line)
        q_pid.append(line['pid'])

for pid in q_pid:
    if pid not in exist_pid:
      file_name = f"{pid}.json"
      file_path = os.path.join(INPUT_FOLDER, file_name)
      
      try:
          file_content = json.load(open(file_path, "r"))
      except FileNotFoundError:
          print(f"File {file_name} not found. Skipping...")
          continue
      
      for c_key in file_content:
          length = len(file_content[c_key].split("."))
          if length >= 5:
            # print(file_content[c_key])
            GENERAL_QUESTION = f"""
              Question: Describe the patient personal information.
              Question: Describe the patient experience.
              Question: Did you notice any symptoms, such as a fever, cough, or respiratory issues?
              Question: What's the diagnosis?
              Question: What's the direct evidence that points to this diagnosis?
              Question: What’s the imaging (only provided the figure explanation here) suggest?
              Question: What’s the examination suggest?
              Question: Is there any suggestion?

              Please only output the complete sentences in the provided text that correspond to the above questions. You can list several sentences related to my query and classify which answer belongs to which query.
              Mention: Please do not miss any information in the provided text, and all of your answers should exactly match the provided text; do not change any symbols. 
              If there no information about the question just reply '$No$'
              Format should be:
              “Question: \nAnswer: \n\nQuestion: \nAnswer: ”
              {file_content[c_key]}
              """
            inputs = tokenizer(GENERAL_QUESTION, return_tensors="pt")
            input_ids = inputs['input_ids'].to("cuda")
            attention_mask = inputs['attention_mask'].to("cuda")
            output_ids = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=1000, num_return_sequences=1, temperature=0.7)

            output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            response = output_text.replace(GENERAL_QUESTION, "")

            pattern = r"Question: (.*?)\nAnswer: (.*?)\n"
            matches = re.findall(pattern, response)

            qa_pairs = [{"Question": match[0], "Answer": match[1]} for match in matches]
            answer_lst = [match[1] for match in matches if "No" not in match[1]]
            if len(answer_lst) == 8:
                with open(OUTPUT_FILE, "a") as f_out:
                    json.dump({"pid": pid, "case_content": file_content[c_key], "response": response}, f_out)
                    f_out.write("\n")
            
            # user_decision = input("Do you pass this case? (y/n/exit): ").strip().lower()
            # if user_decision == 'n':
            #     with open(OUTPUT_FILE, "a") as f_out:
            #         json.dump({"pid": pid, "case_content": file_content[c_key], "response": response}, f_out)
            #         f_out.write("\n")
            #     print("Case not passed and saved. Moving to the next.")
            # elif user_decision == 'exit':
            #     print("Exiting the process.")
            #     exit()
            # else:
            #     print("Case passed. Moving to the next.")
