import pandas as pd

size = "small"

print("---------------------- SCRIPT GOOD/BAD OUTPUT ANALYSIS ----------------------")

# ix = df["response_type"].str.contains("4_options") # select mcq
bad_output = pd.read_csv(f"bad_output_samples_{size}_models_marked_w_chatgpt.csv")#[ix]

# ix = df["response_type"].str.contains("4_options") # select mcq
good_output = pd.read_csv(f"good_output_samples_{size}_models_marked_w_chatgpt.csv")#[ix]

print("LEN GOOD:", len(good_output))
print("LEN BAD:", len(bad_output))
print("LEN TOTAL:", len(good_output)+len(bad_output))

# print("MODELS:", good_output["model"].nunique(), bad_output["model"].nunique())

models = set()
[models.add(x) for x in good_output["model"].unique()]
[models.add(x) for x in bad_output["model"].unique()]

print("TOTAL MODELS:", len(models))

good_output["correct"] = good_output["correct"].replace({3:0})
print(good_output["correct"].value_counts())
print(good_output["correct"].value_counts(normalize=True)*100)

bad_output["correct"] = bad_output["correct"].replace({3:1})
print(bad_output["correct"].value_counts())
print(bad_output["correct"].value_counts(normalize=True)*100)

df = pd.concat([good_output, bad_output])
print(df["correct"].value_counts())
print(df["correct"].value_counts(normalize=True)*100)

""" MCQ
LEN GOOD: 1081
LEN BAD: 728
TOTAL MODELS: 26

# GOOD
1    995
0     86

1    92.044403
0     7.955597

# BAD
1    721
0      7

1    99.038462
0     0.961538

# COMBINED
1    1716
0      93

1    94.859038
0     5.140962

"""

print("---------------------- ChatGPT GOOD/BAD OUTPUT ANALYSIS ----------------------")

print(good_output["correct_chatgpt"].value_counts())
print(good_output["correct_chatgpt"].value_counts(normalize=True)*100)

print(bad_output["correct_chatgpt"].value_counts())
print(bad_output["correct_chatgpt"].value_counts(normalize=True)*100)

print(df["correct_chatgpt"].value_counts())
print(df["correct_chatgpt"].value_counts(normalize=True)*100)


""" MCQ
## GOOD
1.0    707
0.0    374

1.0    65.402405
0.0    34.597595

## BAD
1.0    494
0.0    234

1.0    67.857143
0.0    32.142857

## COMBINED
1.0    1201
0.0     608

1.0    66.390271
0.0    33.609729
"""