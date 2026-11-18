import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:\\Users\\daksh chauhan\\OneDrive\\python\\PREDICTIVE PROJECT\\Vet_Clinical_Dataset.csv")

plt.figure(figsize=(10, 6))

sns.histplot(data=df, x="weight",bins=25, kde=True)

plt.title("Distribution of Animal Weight", fontsize=16)
plt.xlabel("Weight", fontsize=12)
plt.ylabel("Number of Animals", fontsize=12)
plt.tight_layout()
plt.show()

#Boxplot — Temperature outliers
plt.figure(figsize=(10, 4))
sns.boxplot(data=df, x="temperature", width=0.4)
plt.title("Temperature Distribution and Outliers", fontsize=16)
plt.xlabel("Temperature", fontsize=12)
plt.tight_layout()
plt.show()

#Bar plot — Species distribution

plt.figure(figsize=(9, 5))

ax = sns.countplot(data=df, x="species")

for container in ax.containers:
    ax.bar_label(container, fontsize=11)

plt.title("Distribution of Animal Species", fontsize=16)
plt.xlabel("Species", fontsize=12)
plt.ylabel("Number of Records", fontsize=12)
plt.tight_layout()
plt.show()

#Pie chart — Sex distribution
plt.figure(figsize=(8, 7))

df["sex"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%",
    startangle=90,
    textprops={"fontsize": 12}
)

plt.title("Distribution of Animals by Sex", fontsize=16)
plt.ylabel("")
plt.tight_layout()
plt.show()


#Scatter plot — Weight vs temperature

plt.figure(figsize=(10, 6))
sns.scatterplot( data=df, x="weight",y="temperature",hue="species",alpha=0.7,s=60)
sns.regplot( data=df,x="weight",y="temperature",scatter=False,color="red",line_kws={"linewidth": 2}) 
plt.title("Weight vs Temperature with Trend Line", fontsize=16)
plt.xlabel("Weight", fontsize=12)
plt.ylabel("Temperature", fontsize=12)
plt.legend(title="Species")
plt.tight_layout()
plt.show()

#Correlation heatmap — Numerical features

numeric_df = df.select_dtypes(include="number")
plt.figure(figsize=(11, 8))
sns.heatmap(numeric_df.corr(),annot=True,fmt=".2f",cmap="coolwarm",center=0,linewidths=0.5,annot_kws={"size": 10})
plt.title("Correlation Between Numerical Features", fontsize=16)
plt.tight_layout()
plt.show()