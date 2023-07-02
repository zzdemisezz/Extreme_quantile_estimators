import pandas as pd


MODEL_FILENAME = "2023-06-03_19-58-42"

df_summary = pd.read_csv("_config_summary.csv", sep=";")  # associated csv row
df_summary.drop(df_summary.index[df_summary["model_filename"].str.startswith(MODEL_FILENAME)], inplace=True)
df_summary.dropna(axis=0, how="all", inplace=True)
df_summary.to_csv("_config_summary.csv", header=True, index=False, sep=";")