import os

# Configuration
USE_FLASH_ATTENTION = os.getenv("USE_FLASH_ATTENTION", "false").lower() == "true"

# Available models will be populated at runtime
available_models = {}

# Model instances will be shared across routes
model_instances = {}
