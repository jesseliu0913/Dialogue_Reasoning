# Model Configuration

## Parameters

- **model**: `hf` or `openai`  
  Specify the type of model to use.

- **model_args**:  
  For `hf`: Provide the model weights.  
  For `openai`: Specify the model type.

- **task_name**:  
  Indicate the model used, e.g., `llama`, `gpt35`, `gpt4o`, `meditron`, etc.

- **lora_weight**:  
  Load the LoRA weight (e.g., from HuggingFace).

- **output_path**:  
  Specify the output folder path for results.

- **confusion_level**:  
  Set the number of troublemakers, options include `3`, `5`, `10`, etc.

- **task_type**:  
  Choose between `one_round` or `multi_round`.

- **num_fewshot**:  
  Applicable only for `one_round` tasks. Specify the number of few-shot examples.

- **limit**:  
  Optional. To test with a limited dataset, set the number of data points here.

## Example Usage

```yaml
model: hf
model_args: meta-llama/Llama-3.1-70B-Instruct
task_name: llama
lora_weight: JesseLiu/llama_dialogue
output_path: ./output/
confusion_level: 5
task_type: one_round
num_fewshot: 3
limit: 1
