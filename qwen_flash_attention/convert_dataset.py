import json

def convert_dataset():
    # Read the original dataset
    with open('LLaMA-Factory/data/rust_dataset.json', 'r') as f:
        data = json.load(f)

    # Convert to alpaca format
    converted_data = []
    for item in data:
        for conv in item['conversation']:
            converted_item = {
                'instruction': conv['input'],
                'input': '',  # No separate input in our case
                'output': conv['output'],
                'system': conv['system']
            }
            converted_data.append(converted_item)

    # Write the converted dataset
    with open('LLaMA-Factory/data/rust_dataset_converted.json', 'w') as f:
        json.dump(converted_data, f, indent=2)

if __name__ == '__main__':
    convert_dataset()
