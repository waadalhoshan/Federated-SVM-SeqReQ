
import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    df['label_enc'] = df['Label'].map({'Non-Security': 0, 'Security': 1})
    return df
