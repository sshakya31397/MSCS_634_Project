MSCS 634 - Final Project Report
Title: Heart Disease Analysis Using Machine Learning
Deliverables: Data Cleaning & EDA | Regression | Classification, Clustering & Pattern Mining

📁 Dataset Overview
We used a modified version of the UCI Heart Disease dataset with 1,035 records and 14 attributes including age, sex, chest pain type, cholesterol, and more. Categorical attributes were one-hot encoded, and numerical features were cleaned and normalized where necessary.

Depending on the task:

Target (for classification and clustering): target (1 = heart disease, 0 = no disease)
Target (for regression): chol (serum cholesterol level)

🔹 Deliverable 1: Data Collection, Cleaning, and Exploration

📊 Data Cleaning Steps

Missing Values:
Found in chol, thalach, and oldpeak
Filled using median imputation

Duplicate Rows:
Removed using drop_duplicates()

Outliers:
Detected extreme values in chol, trestbps, thalach, oldpeak
Capped using the IQR method

📈 Exploratory Data Analysis (EDA)

Feature Distributions: Validated for normality and shape

Correlation Heatmap:
cp positively correlated with target
thalach and oldpeak negatively correlated

Boxplots: Outlier handling confirmed visually

💡 Key Insights

Dataset is now clean, consistent, and ready for modeling
Key features for modeling: cp, thalach, oldpeak, exang

Dataset supports regression, classification, and clustering tasks


🔹 Deliverable 2: Regression Modeling and Performance Evaluation

Objective:

Predict cholesterol level (chol) using regression models.

Feature Engineering:
One-hot encoding of categorical variables

drop_first=True used to avoid dummy variable trap

Models Used:
Linear Regression
Ridge Regression (L2 regularization)

Data Split:
80% Train / 20% Test

5-fold Cross-Validation applied

Metrics Used:
MSE, RMSE, R² Score

Results Summary:
Model	R² Score	RMSE	Cross-Val MSE	Notes
Linear Regression	Low	High	Higher	Weak prediction power
Ridge Regression	Slightly Better	Lower	Improved	Best overall performer


🔍 Insight:

Ridge performed better due to regularization
Predicting chol with current features has limited success
Future work: Try different targets or apply feature selection

🔹 Deliverable 3: Classification, Clustering, and Pattern Mining

1️⃣ Classification
Objective:
Predict heart disease presence (target)

Models Used:
Decision Tree Classifier

K-Nearest Neighbors (K=5)

Model	Accuracy	Notes
Decision Tree	~83%	Interpretable, good baseline
KNN (k=5)	~86%	Best after feature standardizing

Hyperparameter Tuning:
Decision Tree: Best max_depth = 4

KNN: Best k = 5

🔍 Insight:

KNN slightly better post-scaling
F1-score and confusion matrix used for evaluation

2️⃣ Clustering (KMeans)
Applied KMeans (k=2) on standardized data

Used PCA to visualize clusters in 2D

🔍 Insight:
Clusters roughly matched true labels, confirming natural separability despite being unsupervised.

3️⃣ Association Rule Mining (Apriori)
Used Apriori on one-hot encoded features (cp, fbs, etc.)

Discovered strong rules linking symptoms and disease

Example Rule:
If cp_2 is present → target = 1

Confidence: 78.5%

Lift: 1.53

🔍 Insight:
Uncovered meaningful symptom combinations linked to heart disease

Helpful for rule-based health decision systems

🧠 Final Reflections
Challenges Faced:
Data issues like missing values, outliers, and noisy records

Need for proper encoding and feature scaling

Parameter tuning for best performance

Setting appropriate thresholds in pattern mining

Tools & Libraries:
Python: Pandas, Scikit-learn, Seaborn, Matplotlib



✅ Conclusion
Across all deliverables, we successfully:
Cleaned and explored a complex health dataset
Built and evaluated regression models for cholesterol prediction
Applied classification, clustering, and pattern mining to understand disease presence
Identified key insights and limitations, laying a foundation for further analysis in healthcare applications

This project demonstrates the versatility of machine learning in uncovering patterns and supporting diagnostic predictions in the medical field.