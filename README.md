# 🐾 Veterinary Health & Parasitology Analysis Using Machine Learning

## 📌 Project Overview

This project focuses on analyzing veterinary clinical data and predicting different veterinary diagnoses using Machine Learning techniques.

The project uses Python for data preprocessing, Exploratory Data Analysis (EDA), data visualization, and implementation of multiple classification algorithms.

The main objective is to explore veterinary health patterns and compare the performance of different machine learning models.

## 🎯 Project Objectives

- Perform data cleaning and preprocessing.
- Conduct Exploratory Data Analysis (EDA).
- Analyze relationships between different clinical features.
- Create meaningful data visualizations.
- Implement multiple Machine Learning classification algorithms.
- Evaluate and compare model performance.

## 📂 Dataset

**Dataset:** Veterinary Health and Parasitology Dataset

**Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/blaineee/veterinary-health-and-parasitology-dataset)

The dataset contains veterinary clinical information and 13 diagnosis categories.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine Learning |

## 📊 Exploratory Data Analysis (EDA)

The following analyses were performed:

- Weight Distribution Analysis
- Temperature Distribution Analysis
- Species Distribution Analysis
- Gender Distribution Analysis
- Weight vs Temperature Relationship
- Correlation Heatmap

## 🤖 Machine Learning Algorithms

Five classification algorithms were implemented:

1. Decision Tree
2. Random Forest
3. Gaussian Naive Bayes
4. K-Nearest Neighbors (KNN)
5. Support Vector Machine (SVM)

## 📈 Model Performance

The following accuracy scores were obtained during the experiments:

| Machine Learning Model | Accuracy |
|---|---:|
| Random Forest | 100.00% |
| Support Vector Machine (SVM) | 63.25% |
| Decision Tree | 57.75% |
| Gaussian Naive Bayes | 48.75% |
| K-Nearest Neighbors (KNN) | 46.75% |

**Note:** These are experimental results from the current model runs. The 100% Random Forest accuracy should be investigated further to check for possible data leakage, duplicate records, or overlap between training and testing data.

## 📋 Model Evaluation Metrics

The models were evaluated using:

- Accuracy Score
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

## 📁 Project Structure

```text
Veterinary-Health-ML-Analysis/
│
├── Dataset/
│   └── Vet_Clinical_Dataset.csv
│
├── EDA/
│   └── Exploratory_Data_Analysis.py
│
├── Models/
│   ├── Decision_tree.py
│   ├── Random_forest.py
│   ├── Naive_bayes.py
│   ├── KNN.py
│   └── SVM.py
│
├── Images/
│   └── EDA and Model Results
│
└── README.md
```

*Adjust the folder and file names above to match your actual repository structure.*

## ⚙️ Installation and Usage

**1. Clone the repository**

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
```

**2. Navigate to the project directory**

```bash
cd Veterinary-Health-ML-Analysis
```

**3. Install required libraries**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

**4. Run the Python scripts**

```bash
python SVM.py
python KNN.py
python "Naive bayes.py"
python "Random forest.py"
python "Decision tree.py"
```

Make sure the dataset path in each script matches your local folder structure.

## 💡 Key Learnings

- Understanding and performing EDA on real-world datasets.
- Data cleaning and preprocessing using Pandas.
- Creating visualizations to identify patterns.
- Implementing different classification algorithms.
- Evaluating machine learning models.
- Comparing model performance using multiple evaluation metrics.

## 👨‍💻 Author

**Daksh Chauhan**

B.Tech – Computer Science and Engineering  
Lovely Professional University

## 📜 Disclaimer

This project was developed for academic and learning purposes. The model predictions are experimental and should not be used as a substitute for professional veterinary diagnosis.

---

⭐ If you find this project interesting, feel free to explore the repository!
