import torch

def check_gpu():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i + 1}:")
            print(f"  Name: {gpu_props.name}")
            print(f"  Total Memory: {gpu_props.total_memory / (1024**3):.1f} GB")
            print(f"  Compute Capability: {gpu_props.major}.{gpu_props.minor}")
    else:
        print("No CUDA GPU available")

if __name__ == "__main__":
    check_gpu()
