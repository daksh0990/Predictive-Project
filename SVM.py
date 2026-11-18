import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df = pd.read_csv("C:\\Users\\daksh chauhan\\OneDrive\\python\\PREDICTIVE PROJECT\\Vet_Clinical_Dataset.csv")

# 2. Remove rows with missing target
df = df.dropna(subset=["definitive diagnosis"])

# 3. Separate features and target
X = df.drop(columns=["definitive diagnosis","Patient ID","physical examination notes","additional clinical notes"], errors="ignore")
y = df["definitive diagnosis"]

# 4. Convert categorical features into numbers
X = pd.get_dummies(X, dummy_na=True)

# 5. Handle missing values
imputer = SimpleImputer(strategy="median")
X = pd.DataFrame( imputer.fit_transform(X),columns=X.columns,index=X.index)

# 6. Encode target labels
le = LabelEncoder()
y = le.fit_transform(y)

# 7. Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 8. Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 9. Train SVM model
model = SVC(kernel="rbf", random_state=42)
model.fit(X_train, y_train)

# 10. Predict
y_pred = model.predict(X_test)

# 11. Evaluate model
print("SVM Accuracy:",round(accuracy_score(y_test, y_pred) * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred,labels=range(len(le.classes_)),target_names=le.classes_,
zero_division=0
))

# 12. Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))