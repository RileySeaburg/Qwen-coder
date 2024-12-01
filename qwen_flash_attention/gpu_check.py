import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_gpu_info():
    """Get GPU information."""
    if not torch.cuda.is_available():
        return None

    device = torch.cuda.current_device()
    gpu_properties = torch.cuda.get_device_properties(device)
    total_memory = gpu_properties.total_memory / (1024 * 1024)  # Convert to MB
    free_memory = torch.cuda.mem_get_info()[0] / (1024 * 1024)  # Convert to MB

    return {
        "name": gpu_properties.name,
        "total_memory": total_memory,
        "free_memory": free_memory,
        "compute_capability": f"{gpu_properties.major}.{gpu_properties.minor}"
    }

def select_best_model():
    """Select the best model based on available GPU memory."""
    gpu_info = get_gpu_info()
    if not gpu_info:
        logger.info("No GPU detected, using CPU model")
        return {
            "name": "qwen2.5-coder-1.5b",
            "repo_id": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "memory_required": 3072  # Reduced memory requirement
        }

    logger.info(f"GPU Information:")
    logger.info(f"Name: {gpu_info['name']}")
    logger.info(f"Total Memory: {gpu_info['total_memory']:.1f} MiB")
    logger.info(f"Free Memory: {gpu_info['free_memory']:.1f} MiB")
    logger.info(f"Compute Capability: {gpu_info['compute_capability']}")

    logger.info(f"\nSelecting best model for {gpu_info['free_memory']:.1f}MB free GPU memory")

    # Define models with reduced memory requirements
    models = [
        {
            "name": "qwen2.5-coder-1.5b",
            "repo_id": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "memory_required": 3072  # Reduced from 4096
        }
    ]

    # Select model based on available memory
    selected_model = None
    for model in models:
        if model["memory_required"] <= gpu_info["free_memory"]:
            selected_model = model
            break

    if not selected_model:
        logger.info("Not enough GPU memory, using smallest model")
        selected_model = models[0]

    logger.info(f"Selected model: {selected_model['name']} (requires {selected_model['memory_required']}MB)")
    return selected_model

if __name__ == "__main__":
    select_best_model()
