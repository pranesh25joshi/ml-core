import numpy as np
import pandas as pd

df = pd.read_csv("10naive-bayes/play_tennis.csv")

print(df.head())
print("\n---------------------\n")

print(df['play'].value_counts()["Yes"])
p_yes = df['play'].value_counts()["Yes"] / len(df)
print(f"Probability of Yes: {p_yes}")

p_no = df['play'].value_counts()["No"] / len(df)
print(f"Probability of No: {p_no}")

print("\n---------------------\n")

print(pd.crosstab(df['outlook'], df['play']))
print("\n---------------------\n")

print(pd.crosstab(df['temp'], df['play']))
print("\n---------------------\n")
print(pd.crosstab(df['humidity'], df['play']))
print("\n---------------------\n")
print(pd.crosstab(df['wind'], df['play']))

p_sunny_yes = 2 / 9
p_sunny_no = 3 / 5

p_hot_yes = 2 / 9
p_hot_no = 2 / 5


p_high_yes = 3 / 9
p_high_no = 4 / 5


p_weak_yes = 6 / 9
p_weak_no = 2 / 5

p_yes_query = p_yes * p_sunny_yes * p_hot_yes * p_high_yes * p_weak_yes
p_no_query = p_no * p_sunny_no * p_hot_no * p_high_no * p_weak_no

# Maximum A Posteriori
print(f"Score for Yes: {p_yes_query}")
print(f"Score for No: {p_no_query}")

if p_yes_query > p_no_query:
    print("Prediction: Yes")
else:
    print("Prediction: No")