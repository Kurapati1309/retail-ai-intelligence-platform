import pandas as pd
from sklearn.model_selection import train_test_split

def stratified_split(df: pd.DataFrame, label_col: str, test_size=0.2, random_state=42):
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[label_col] if label_col in df.columns else None
    )
    return train_df, test_df
