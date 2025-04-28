import json
import argparse
import re
import html

def load_jsonl(file_path):
    """Load JSONL file and return a list of records."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def clean_text(text):
    """Clean text to avoid import issues."""
    text = text.replace('\n\n', '\n')
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text.strip()

import re

def clean_dialogue(text):
    if "**" in text:
        return text.split("**", 1)[1]
    return text


def generate_unified_mc_questions(input_file, output_file, num_samples=150):
    """Generate MC questions with HTML to keep everything as one question."""
    data = load_jsonl(input_file)
    
    # Sample records if there are more than requested
    if len(data) > num_samples:
        data = data[:num_samples]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write block header
        f.write("[[Block:Dialogue Evaluation]]\n\n")
        
        # Process each entry
        for i, entry in enumerate(data):
            qid = i + 1
            
            # Format the paragraph and dialogue text
            paragraph = clean_text(entry['input'])
            dialogue = clean_text(entry['response'])
            dialogue = clean_dialogue(dialogue)
            
            
            # Escape HTML entities
            paragraph = html.escape(paragraph)
            dialogue = html.escape(dialogue)
            
            # Create a unified question with HTML formatting
            question_text = f"""To what extent do you think the dialogue captures all the essential information from the paragraph?<br><br>
<strong>Paragraph:</strong><br>
{paragraph}<br><br>
<strong>Dialogue:</strong><br>
>>{dialogue}"""
            
            # Write question with export tag
            f.write(f"MC{qid}. {question_text}\n\n")
            
            # Write choices
            f.write("4 - Fully Covered: All essential information is present and clearly conveyed.\n")
            f.write("3 - Mostly Covered: Most key ideas are included, with minor omissions\n")
            f.write("2 - Moderately Covered: About half of the vital information is included.\n")
            f.write("1 - Minimally Covered: Only a few important points are mentioned.\n")
            f.write("0 - Not Covered: The dialogue misses or distorts the core information.\n\n")
            
            # Add page break between questions except for the last one
            if i < len(data) - 1:
                f.write("[[PageBreak]]\n\n")
    
    print(f"Created file: {output_file} with {len(data)} unified MC questions")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate HTML-formatted MC questions for Qualtrics from a JSONL file.')
    parser.add_argument('--input', required=True, help='Path to input JSONL file')
    parser.add_argument('--output', required=True, help='Output path for the generated TXT file')
    parser.add_argument('--samples', type=int, default=150, help='Number of samples to include (default: 150)')
    
    args = parser.parse_args()
    
    generate_unified_mc_questions(args.input, args.output, args.samples)
# python generate_dialogue_evaluation_survey.py --input /playpen/jesse/Dialogue_Reasoning/dialogue_generation/dialogue_set/dialogue_0-5000_gemini.jsonl --output dialogue_eval.txt --samples 150
# python generate_dialogue_evaluation_survey.py --input /playpen/jesse/Dialogue_Reasoning/dialogue_generation/dialogue_medqa_gemini.jsonl --output dialogue_eval3.txt --samples 150