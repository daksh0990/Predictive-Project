import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df = pd.read_csv("C:\\Users\\daksh chauhan\\OneDrive\\python\\PREDICTIVE PROJECT\\Vet_Clinical_Dataset.csv")

# 2. Remove rows with missing target
df = df.dropna(subset=["definitive diagnosis"])

# 3. Separate features and target
X = df.drop(columns=["definitive diagnosis","Patient ID","physical examination notes", "additional clinical notes"], errors="ignore")

y = df["definitive diagnosis"]

# 4. Convert categorical features into numbers
X = pd.get_dummies(X, dummy_na=True)

# 5. Encode target labels
le = LabelEncoder()
y = le.fit_transform(y)

# 6. Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 7. Handle missing values
imputer = SimpleImputer(strategy="median")

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# 8. Create Random Forest model
model = RandomForestClassifier( n_estimators=100, random_state=42)

# 9. Train model
model.fit(X_train, y_train)

# 10. Predict
y_pred = model.predict(X_test)

# 11. Evaluate model
print("Random Forest Accuracy:",round(accuracy_score(y_test, y_pred) * 100, 2), "%")

print("\nClassification Report:")
print(classification_report( y_test, y_pred,labels=range(len(le.classes_)),target_names=le.classes_,zero_division=0
))

# 12. Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))