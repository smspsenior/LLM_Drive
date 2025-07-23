import pandas as pd
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
from tqdm import tqdm
import argparse
from pathlib import Path

"""
This script generates dataset for training. 
In particular, it runs two VLM to inference possible actions given front RGB and semantic bev, respectively. 
"""

VLM_MODEL_NAME = "llava-hf/llava-1.5-7b-hf"

def get_llm_explanation(processor, model, image_path, prompt):
    try:
        raw_image = Image.open(image_path).convert('RGB')
        inputs = processor(text=prompt, images=raw_image, return_tensors="pt").to(model.device, torch.float16)
        output = model.generate(**inputs, max_new_tokens=75)
        response_text = processor.decode(output[0], skip_special_tokens=True)
        
        if 'ASSISTANT:' in response_text:
            clean_text = response_text.split('ASSISTANT:')[-1].strip()
        else:
            clean_text = "[ERROR] Could not parse model output"
        return clean_text
        
    except Exception as e:
        print(f"Caught an exception while processing {image_path}: {e}")
        return "[ERROR] Could not generate explanation"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate explanations for a given dataset index CSV using a VLM. ")
    parser.add_argument("input_csv", help="Path to the input CSV file (e.g., ./split_datasets/part_1.csv). ")
    parser.add_argument("-o", "--output_csv", help="Path for the output CSV file (optional).")
    args = parser.parse_args()
    input_path = Path(args.input_csv)
    
    if args.output_csv:
        output_path = Path(args.output_csv)
    else:
        output_path = input_path.parent / f"explanations_{input_path.name}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model {VLM_MODEL_NAME} to {device}...")
    model = LlavaForConditionalGeneration.from_pretrained(
        VLM_MODEL_NAME,
        torch_dtype=torch.float16,
        load_in_4bit=True
    )
    processor = AutoProcessor.from_pretrained(VLM_MODEL_NAME)
    print("Model loaded successfully!")

    df_index = pd.read_csv(input_path)
    results = []
    print(f"Starting to generate explanations for {len(df_index)} frames from '{input_path.name}'...")

    for index, row in tqdm(df_index.iterrows(), total=df_index.shape[0], desc=f"Processing {input_path.name}"):
        prompt1 = "USER: <image>\nThis is a front-facing camera view from a car. Describe the road ahead and advise if it's safe to proceed. Be concise.\nASSISTANT:"
        explanation1 = get_llm_explanation(processor, model, row['rgb_path'], prompt1)
        
        prompt2_template = "USER: <image>\nThis is a bird's-eye view semantic map. The initial plan is: '{}'. Based on this top-down view, re-evaluate the safety. Are there any hidden risks from the sides or rear?\nASSISTANT:"
        prompt2 = prompt2_template.format(explanation1)
        explanation2 = get_llm_explanation(processor, model, row['bev_path'], prompt2)
        
        results.append({
            'explanation1': explanation1,
            'explanation2': explanation2,
            'throttle': row['throttle'],
            'brake': row['brake']
        })

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_path, index=False)
    print(f"\nExplanations generated and saved to: {output_path}")
    
