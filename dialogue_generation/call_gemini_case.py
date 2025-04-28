import os
import json
import argparse
import time
import random
import google.generativeai as genai


parser = argparse.ArgumentParser(description="Case Report to Dialogue Based on Gemini")
parser.add_argument('--file_path', type=str, help="which file data you want to use")
parser.add_argument('--start_index', type=int, default=5000, help="Start index for case data")
parser.add_argument('--end_index', type=int, default=10000, help="End index for case data")
args = parser.parse_args()

genai.configure(api_key="AIzaSyDLIzz__3NvdGYC3W8vDMJq-P-5TuqRCAA")

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


with open(args.file_path, 'r') as file:
    total_lines = sum(1 for _ in file)

end_index = min(args.end_index, total_lines - 1)
data = []
with open(args.file_path, 'r') as file:
    for idx, line in enumerate(file):
        if idx >= args.start_index:
            data.append(json.loads(line))
        if idx >= end_index:
            break


output_file = f"./dialogue_set/dialogue_{args.start_index}-{end_index}_gemini.jsonl"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

existing_lines = 0
with open(output_file, "r") as f_read:
    for _ in f_read:
        existing_lines += 1

with open(output_file, "a") as f_write:
    for idx, line in enumerate(data[existing_lines:], start=existing_lines):
        line_dict = {}
        input_text = f"{line['case']}"
        prefix = """Please convert the following paragraph into a doctor-patient dialogue. Ensure that all the information provided, like personal details, symptoms, examination findings, diagnosis, and treatment, is included. Use natural conversational language to connect the details, but do not introduce any new information. The dialogue should not be too redundant:"""
        prompt = f"{prefix}\n{input_text}"
        
        print(f"Processing case {idx + args.start_index}/{end_index}...")
        response = call_gemini(prompt)
        
        line_dict['pid'] = line['pid']
        line_dict['input'] = input_text
        line_dict['response'] = response
        f_write.write(json.dumps(line_dict) + "\n")
        
        time.sleep(1)

print(f"Conversion complete. Output saved to {output_file}")

# nohup python call_gemini_case.py --file_path "/playpen/jesse/Dialogue_Reasoning/case_report/output/tuning/raw_case2.jsonl" > case2dialogue.log 2>&1 &