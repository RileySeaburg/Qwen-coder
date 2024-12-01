import torch
import psutil
from typing import Dict, List

class DetailedGPUChecker:
    def __init__(self):
        self.compatible_models = {
            # Models that can run on consumer GPUs
            "Qwen/Qwen-1_8B": {
                "vram": 4,
                "ram": 8,
                "compute_capability": 7.0
            },
            "Qwen/Qwen-7B": {
                "vram": 7,
                "ram": 16,
                "compute_capability": 7.0
            },
            "THUDM/chatglm2-6b": {
                "vram": 6,
                "ram": 16,
                "compute_capability": 7.0
            },
            "THUDM/chatglm3-6b": {
                "vram": 6,
                "ram": 16,
                "compute_capability": 7.0
            },
            "microsoft/phi-2": {
                "vram": 4,
                "ram": 8,
                "compute_capability": 7.0
            }
        }

    def get_detailed_gpu_info(self) -> Dict:
        """Get detailed information about the GPU."""
        if not torch.cuda.is_available():
            return {}

        gpu_info = {}
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_info[i] = {
                "name": props.name,
                "vram_total": props.total_memory / (1024**3),  # GB
                "compute_capability": f"{props.major}.{props.minor}",
                "multi_processor_count": props.multi_processor_count,
                "cuda_cores": props.multi_processor_count * 64  # Approximate for most architectures
            }
            
            # Get free VRAM
            free_mem, total_mem = torch.cuda.mem_get_info(i)
            gpu_info[i]["vram_free"] = free_mem / (1024**3)  # GB
            
        return gpu_info

    def get_system_info(self) -> Dict:
        """Get detailed system information."""
        return {
            "ram_total": psutil.virtual_memory().total / (1024**3),  # GB
            "ram_available": psutil.virtual_memory().available / (1024**3),  # GB
            "cpu_count": psutil.cpu_count(),
            "cpu_freq": psutil.cpu_freq().max if psutil.cpu_freq() else None
        }

    def find_compatible_models(self, gpu_info: Dict, system_info: Dict) -> Dict[str, List[str]]:
        """Find models that can run on the current hardware."""
        results = {
            "recommended_models": [],
            "possible_models": [],
            "warnings": []
        }

        if not gpu_info:
            results["warnings"].append("No NVIDIA GPU detected!")
            return results

        gpu = gpu_info[0]  # Using first GPU
        compute_cap = float(gpu["compute_capability"])
        
        for model, requirements in self.compatible_models.items():
            if (compute_cap >= requirements["compute_capability"] and 
                gpu["vram_total"] >= requirements["vram"] and 
                system_info["ram_total"] >= requirements["ram"]):
                
                # Check if it's a good fit or just barely meets requirements
                if (gpu["vram_total"] >= requirements["vram"] * 1.5 and 
                    system_info["ram_total"] >= requirements["ram"] * 1.5):
                    results["recommended_models"].append(model)
                else:
                    results["possible_models"].append(model)

        return results

    def print_detailed_report(self):
        """Print a detailed system report with specific model recommendations."""
        print("=== Detailed Hardware Analysis ===\n")
        
        gpu_info = self.get_detailed_gpu_info()
        system_info = self.get_system_info()
        
        # Print GPU Information
        if gpu_info:
            for i, gpu in gpu_info.items():
                print(f"GPU {i + 1} Specifications:")
                print(f"  Model: {gpu['name']}")
                print(f"  VRAM: {gpu['vram_total']:.1f}GB")
                print(f"  Free VRAM: {gpu['vram_free']:.1f}GB")
                print(f"  Compute Capability: {gpu['compute_capability']}")
                print(f"  CUDA Cores: {gpu['cuda_cores']}")
                print(f"  Multi-Processors: {gpu['multi_processor_count']}")
        
        # Print System Information
        print("\nSystem Specifications:")
        print(f"  Total RAM: {system_info['ram_total']:.1f}GB")
        print(f"  Available RAM: {system_info['ram_available']:.1f}GB")
        print(f"  CPU Cores: {system_info['cpu_count']}")
        if system_info['cpu_freq']:
            print(f"  CPU Max Frequency: {system_info['cpu_freq']:.1f}MHz")
        
        # Print Model Compatibility
        results = self.find_compatible_models(gpu_info, system_info)
        print("\n=== Model Recommendations ===\n")
        
        if results["recommended_models"]:
            print("Recommended Models (Optimal Performance):")
            for model in results["recommended_models"]:
                print(f"✓ {model}")
                print(f"  Required VRAM: {self.compatible_models[model]['vram']}GB")
                print(f"  Required RAM: {self.compatible_models[model]['ram']}GB")
        
        if results["possible_models"]:
            print("\nPossible Models (May have performance limitations):")
            for model in results["possible_models"]:
                print(f"? {model}")
                print(f"  Required VRAM: {self.compatible_models[model]['vram']}GB")
                print(f"  Required RAM: {self.compatible_models[model]['ram']}GB")
        
        if results["warnings"]:
            print("\nWarnings:")
            for warning in results["warnings"]:
                print(f"! {warning}")

if __name__ == "__main__":
    checker = DetailedGPUChecker()
    checker.print_detailed_report()
