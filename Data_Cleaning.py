import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:\\Users\\daksh chauhan\\OneDrive\\python\\PREDICTIVE PROJECT\\Vet_Clinical_Dataset.csv")

print("Shape:", df.shape)
print(df.head())
print(df.describe())

#checking missing values
missing = pd.DataFrame({
    "Missing Count": df.isnull().sum(),
    "Missing %": (df.isnull().mean() * 100).round(2)
})

#Checking Duplicate Rows
print("Duplicate rows:", df.duplicated().sum())

# Remove duplicate rows
df.drop_duplicates(inplace=True)

print("Duplicate rows removed!")
print("Dataset shape:", df.shape)

# Check missing values before cleaning
print("Missing values before cleaning:")
print(df.isnull().sum())

# Fill numerical missing values with median
num_cols = df.select_dtypes(include=np.number).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())


