import torch
import psutil
import subprocess
import sys
from typing import Dict, List, Tuple

class SystemChecker:
    # Model requirements (approximate values in GB)
    MODEL_REQUIREMENTS = {
        "nvidia/h100-opt-6.7b": {
            "vram": 14,
            "ram": 16,
            "compute_capability": 8.0
        },
        "nvidia/h100-opt-13b": {
            "vram": 28,
            "ram": 32,
            "compute_capability": 8.0
        },
        "nvidia/h100-opt-30b": {
            "vram": 60,
            "ram": 64,
            "compute_capability": 8.0
        },
        "nvidia/h100-opt-66b": {
            "vram": 120,
            "ram": 128,
            "compute_capability": 8.0
        }
    }

    @staticmethod
    def get_gpu_info() -> List[Dict]:
        """Get information about available NVIDIA GPUs."""
        if not torch.cuda.is_available():
            return []

        gpus = []
        for i in range(torch.cuda.device_count()):
            gpu = {
                "name": torch.cuda.get_device_name(i),
                "vram": torch.cuda.get_device_properties(i).total_memory / (1024**3),  # Convert to GB
                "compute_capability": float(f"{torch.cuda.get_device_capability(i)[0]}.{torch.cuda.get_device_capability(i)[1]}")
            }
            gpus.append(gpu)
        return gpus

    @staticmethod
    def get_system_ram() -> float:
        """Get total system RAM in GB."""
        return psutil.virtual_memory().total / (1024**3)

    def check_system(self) -> Dict[str, List[str]]:
        """Check which models can run on the system."""
        gpus = self.get_gpu_info()
        system_ram = self.get_system_ram()
        
        results = {
            "supported_models": [],
            "unsupported_models": [],
            "warnings": []
        }

        if not gpus:
            results["warnings"].append("No NVIDIA GPU detected! Cannot run any models.")
            results["unsupported_models"] = list(self.MODEL_REQUIREMENTS.keys())
            return results

        for model, requirements in self.MODEL_REQUIREMENTS.items():
            can_run = True
            warnings = []

            # Check GPU requirements
            max_vram = max(gpu["vram"] for gpu in gpus)
            max_compute = max(gpu["compute_capability"] for gpu in gpus)
            
            if max_compute < requirements["compute_capability"]:
                can_run = False
                warnings.append(f"Insufficient compute capability: {max_compute} < {requirements['compute_capability']}")
            
            if max_vram < requirements["vram"]:
                can_run = False
                warnings.append(f"Insufficient VRAM: {max_vram:.1f}GB < {requirements['vram']}GB")

            # Check RAM requirements
            if system_ram < requirements["ram"]:
                can_run = False
                warnings.append(f"Insufficient RAM: {system_ram:.1f}GB < {requirements['ram']}GB")

            if can_run:
                results["supported_models"].append(model)
            else:
                results["unsupported_models"].append(model)
                results["warnings"].extend(warnings)

        return results

    def print_report(self):
        """Print a detailed system report and model compatibility."""
        print("=== System Report ===\n")
        
        # Print GPU Information
        gpus = self.get_gpu_info()
        if gpus:
            print("GPU Information:")
            for i, gpu in enumerate(gpus):
                print(f"GPU {i + 1}:")
                print(f"  Name: {gpu['name']}")
                print(f"  VRAM: {gpu['vram']:.1f}GB")
                print(f"  Compute Capability: {gpu['compute_capability']}")
        else:
            print("No NVIDIA GPU detected!")
        
        # Print RAM Information
        system_ram = self.get_system_ram()
        print(f"\nSystem RAM: {system_ram:.1f}GB")
        
        # Print Model Compatibility
        results = self.check_system()
        print("\n=== Model Compatibility ===\n")
        
        if results["supported_models"]:
            print("Supported Models:")
            for model in results["supported_models"]:
                print(f"✓ {model}")
        
        if results["unsupported_models"]:
            print("\nUnsupported Models:")
            for model in results["unsupported_models"]:
                print(f"✗ {model}")
        
        if results["warnings"]:
            print("\nWarnings:")
            for warning in results["warnings"]:
                print(f"! {warning}")

if __name__ == "__main__":
    checker = SystemChecker()
    checker.print_report()
