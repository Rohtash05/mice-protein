# Mice Protein Expression Classification — Machine Learning Assignment 2

## a. Problem statement
The project compares five supervised classification algorithms to classify eight experimental mouse groups using protein-expression measurements. It demonstrates a complete ML workflow: data retrieval, preprocessing, leakage-aware validation, model evaluation, saving trained models and deployment through Streamlit.

## b. Dataset description
- **Dataset:** Mice Protein Expression
- **Source:** UCI Machine Learning Repository
- **UCI dataset ID:** 342
- **Source URL:** https://archive.ics.uci.edu/dataset/342/mice+protein+expression
- **Instances:** 1,080 measurements
- **Variables:** 77 protein-expression measurements plus metadata and class label (UCI reports 82 features in its dataset listing)
- **Classification classes:** 8 mouse experimental groups
- **Task:** Multiclass classification

The protein measurements are from cerebral-cortex samples. The eight classes represent combinations of genotype, behaviour and treatment. The dataset contains repeated measurements associated with individual mice and missing protein readings.

### Important data-leakage prevention
The `MouseID` field is used only to group repeated measurements during splitting; it is not used as a model input. The metadata columns `Genotype`, `Treatment` and `Behavior` are excluded from the predictors because they define the target class. Only protein-expression measurements are used as input features.

A `StratifiedGroupKFold` split is used so measurements belonging to the same mouse cannot appear in both training and test partitions. Median imputation is fitted within each model pipeline, using training data only. This makes the evaluation more defensible than a random row-wise split that could leak mouse-specific information.

## c. GitHub Repository Link
https://github.com/Rohtash05/mice-protein

## d. Models used
1. Logistic Regression — median imputation, StandardScaler, LogisticRegression (`max_iter=6000`)
2. Decision Tree — median imputation, DecisionTreeClassifier (`max_depth=12`, `min_samples_leaf=3`)
3. K-Nearest Neighbors — median imputation, StandardScaler, KNeighborsClassifier (`n_neighbors=7`, `weights='distance'`)
4. Gaussian Naive Bayes — median imputation, GaussianNB
5. Support Vector Machine — median imputation, StandardScaler, RBF SVC (`C=2.0`, `probability=True`)

### Evaluation metrics
- Accuracy
- Weighted one-vs-rest multiclass ROC AUC
- Weighted precision
- Weighted recall
- Weighted F1 score
- Matthews Correlation Coefficient (MCC)

### Model comparison


| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9861 | 0.9997 | 0.9863 | 0.9861 | 0.9861 | 0.9841 |
| Decision Tree | 0.8843 | 0.9473 | 0.8865 | 0.8843 | 0.8842 | 0.8679 |
| KNN | 0.9769 | 0.9994 | 0.9785 | 0.9769 | 0.9769 | 0.9738 |
| Naive Bayes | 0.8056 | 0.9788 | 0.825 | 0.8056 | 0.8001 | 0.7807 |
| Support Vector Machine | 1 | 1 | 1 | 1 | 1 | 1 |

### Observations on model performance
Complete this section after the experiment. Discuss the actual test metrics and class-wise report:
- **Logistic Regression:** compare weighted F1 and accuracy; explain how a linear classifier performs on protein-expression patterns.
- **Decision Tree:** discuss whether its nonlinear splits generalize to held-out mice and whether its depth constraint affects performance.
- **KNN:** discuss similarity-based classification after scaling high-dimensional protein measurements.
- **Gaussian Naive Bayes:** discuss the conditional-independence assumption and the likely correlations between protein measurements.
- **SVM:** compare its RBF-kernel performance with the other models.
- **Overall winner:** identify the best model using F1, accuracy, AUC and MCC together. Explain any trade-offs.

Write these observations in your own words using your actual BITS Virtual Lab output.

## Streamlit application
The app accepts a labelled test CSV containing the selected protein features and target class. It loads the selected saved model and displays all six metrics, a confusion matrix, classification report, prediction preview and downloadable predictions. The app does not retrain on uploaded test data.

## How to run in BITS Virtual Lab
1. Extract this project.
2. Install dependencies: `pip install -r requirements.txt`
3. Download, preprocess, train and evaluate: `python train_models.py`
4. Confirm that five `.joblib` files appear inside the required singular `model/` directory, and that `test_data.csv`, `model_comparison.csv` and `results/` are generated.
5. Start Streamlit: `streamlit run app.py`
6. Upload `test_data.csv` and test all five models.


## GitHub repository contents
Commit `app.py`, `train_models.py`, `requirements.txt`, `README.md`, `test_data.csv`, `model_comparison.csv`, `model/` and `results/`.

## Streamlit Community Cloud deployment
Push the project to GitHub, create a new Streamlit Community Cloud app, select the repository and `main` branch, set the main file to `app.py`, deploy and test the public URL.

## Dataset citation
Higuera, C., Gardiner, K.J. & Cios, K.J. (2015). Self-organizing maps identify proteins critical to learning in a mouse model of Down syndrome. *PLOS ONE*, 10(6), e0129126. Dataset hosted by the UCI Machine Learning Repository.


