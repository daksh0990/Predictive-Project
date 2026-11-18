import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("C:\\Users\\daksh chauhan\\OneDrive\\python\\PREDICTIVE PROJECT\\Vet_Clinical_Dataset.csv")

# 2. Remove rows with missing target
df = df.dropna(subset=["definitive diagnosis"])

# 3. Separate input and output
X = df.drop(columns=["definitive diagnosis","Patient ID","physical examination notes","additional clinical notes"],errors="ignore")

y = df["definitive diagnosis"]

# 4. Convert categorical features into numbers
X = pd.get_dummies(X, dummy_na=True)

# 5. Fill missing values
imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X),columns=X.columns,index=X.index)

# 6. Encode diagnosis labels
le = LabelEncoder()
y = le.fit_transform(y)

# 7. Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 8. Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 9. Create and train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# 10. Predict diagnoses
y_pred = knn.predict(X_test)

# 11. Evaluate model
print("KNN Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred,labels=range(len(le.classes_)),target_names=le.classes_,zero_division=0))

# 12. Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)