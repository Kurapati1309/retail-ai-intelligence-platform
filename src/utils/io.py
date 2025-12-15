import os
import pandas as pd
from .logging import get_logger

log = get_logger(__name__)

def read_csv(path: str) -> pd.DataFrame:
    log.info(f"Reading: {path}")
    return pd.read_csv(path)

def write_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    log.info(f"Writing: {path}")
    df.to_csv(path, index=False)
