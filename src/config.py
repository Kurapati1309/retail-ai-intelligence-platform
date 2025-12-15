import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Paths:
    project_root: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_raw: str = os.path.join(project_root, "data", "raw")
    data_processed: str = os.path.join(project_root, "data", "processed")
    model_dir: str = os.getenv("MODEL_DIR", os.path.join(project_root, "models_artifacts"))

PATHS = Paths()
