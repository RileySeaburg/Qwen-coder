import json
import os

def convert_dataset():
    print("Current working directory:", os.getcwd())
    input_path = 'LLaMA-Factory/data/rust_dataset.json'
    output_path = 'LLaMA-Factory/data/rust_dataset_converted.json'
    
    print(f"Reading from {input_path}")
    # Read the original dataset
    with open(input_path, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} items from original dataset")

    # Convert to alpaca format
    converted_data = []
    for item in data:
        for conv in item['conversation']:
            # Extract code block from input if present
            input_text = conv['input']
            code_block = ""
            if "```rust" in input_text:
                parts = input_text.split("```rust", 1)
                if len(parts) > 1:
                    code_block = parts[1].split("```", 1)[0]
                    input_text = parts[0].strip()

            converted_item = {
                "instruction": input_text,
                "input": code_block,  # Code block becomes the input
                "output": conv['output'],
                "system": conv['system']
            }
            converted_data.append(converted_item)

    print(f"Writing {len(converted_data)} items to {output_path}")
    # Write the converted dataset
    with open(output_path, 'w') as f:
        json.dump(converted_data, f, indent=2)
    print("Conversion complete")

if __name__ == '__main__':
    convert_dataset()
