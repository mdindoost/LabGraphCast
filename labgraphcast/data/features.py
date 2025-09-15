import numpy as np
import pandas as pd

WEEKDAYS = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

def add_weekday_onehot(df: pd.DataFrame) -> pd.DataFrame:
    if "weekday" not in df.columns:
        df = df.assign(weekday=pd.to_datetime(df["date"]).dt.day_name().str[:3])
    for w in WEEKDAYS:
        df[f"wd_{w}"] = (df["weekday"].str[:3] == w).astype(int)
    return df

def add_slot_posenc(df: pd.DataFrame) -> pd.DataFrame:
    # slot is "HH:MM"
    slot_idx = pd.to_datetime(df["slot"], format="%H:%M").dt.hour + \
               pd.to_datetime(df["slot"], format="%H:%M").dt.minute/60.0
    # normalize to [0, 2π)
    x = 2*np.pi*(slot_idx/24.0)
    df["slot_sin"] = np.sin(x)
    df["slot_cos"] = np.cos(x)
    return df

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_weekday_onehot(df.copy())
    df = add_slot_posenc(df)
    return df
