import numpy as np
import pandas as pd

df = pd.read_csv("10naive-bayes/play_tennis.csv")

print(df.head())

df['play'].value_counts()