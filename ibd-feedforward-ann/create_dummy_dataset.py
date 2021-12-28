import numpy
import pandas as pd 
df = pd.read_csv("data_original.csv")
dum = pd.get_dummies(df)
dum.to_csv("data.csv")

