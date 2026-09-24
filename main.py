import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve
)

from imblearn.over_sampling import SMOTE

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Install XGBoost: pip install xgboost")

#loading Dataset

print("="*70)
print("STEP 1 : LOAD DATASET")
print("="*70)

dataset = pd.read_csv("software_defect_prediction_dataset.csv")

print(dataset.head())
print("\nDataset Shape:", dataset.shape)

dataset.insert(
    0,
    "Test_ID",
    ["TC_" + str(i+1) for i in range(len(dataset))]
)
#HANDLE MISSING VALUES

print("\n"+"="*70)
print("STEP 3 : HANDLE MISSING VALUES")
print("="*70)

dataset.replace("?", np.nan, inplace=True)

for col in dataset.columns:
    if col not in ["Test_ID","defects"]:
        dataset[col]=pd.to_numeric(dataset[col],errors="coerce")

for col in dataset.columns:
    if col not in ["Test_ID","defects"]:
        dataset[col]=dataset[col].fillna(dataset[col].median())

print("\nTotal Missing Values:",dataset.isnull().sum().sum())

dataset["defects"]=(
    dataset["defects"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({"true":1,"false":0})
)
dataset.dropna(subset=["defects"], inplace=True)
dataset["defects"] = dataset["defects"].astype(int)

# ============================================================
# INCREASE DATASET SIZE USING SMOTE
# ============================================================

from imblearn.over_sampling import SMOTE

X = dataset.drop(columns=["Test_ID", "defects"])
y = dataset["defects"]

# Get the majority class size automatically
majority = y.value_counts()[0]

sm = SMOTE(
    sampling_strategy={1: majority},
    random_state=42
)

X_new, y_new = sm.fit_resample(X, y)

# Replace dataset with the augmented dataset
dataset = pd.DataFrame(X_new, columns=X.columns)
dataset["defects"] = y_new
dataset.insert(0, "Test_ID", [f"TC_{i+1}" for i in range(len(dataset))])

# Save the larger dataset
dataset.to_csv("software_defect_prediction_dataset_augmented.csv", index=False)

print("\nAugmented Dataset Shape:", dataset.shape)
print(dataset["defects"].value_counts())

# Continue with your existing code
print("\n"+"="*70)
print("STEP 5 : CLASS DISTRIBUTION")
print("="*70)

defect_data=dataset[dataset["defects"]==1]
non_defect_data=dataset[dataset["defects"]==0]

print("\nTotal Defect Records:",len(defect_data))
print(defect_data.head(10).to_string(index=False))

print("\nTotal Non-Defect Records:",len(non_defect_data))
print(non_defect_data.head(10).to_string(index=False))

X=dataset.drop(columns=["Test_ID","defects"])
y=dataset["defects"]

print("\nSamples:",X.shape[0])
print("Features:",X.shape[1])
# TRAIN TEST SPLIT + SMOTE

print("\n"+"="*70)
print("STEP 7 : TRAIN TEST SPLIT + SMOTE")
print("="*70)

X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nBefore SMOTE")
print(y_train.value_counts())

smote=SMOTE(random_state=42)

X_train,y_train=smote.fit_resample(X_train,y_train)

print("\nAfter SMOTE")
print(y_train.value_counts())
# STRATIFIED CROSS VALIDATION

cv=StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
# HYPERPARAMETER TUNING

print("\n"+"="*70)
print("STEP 9 : HYPERPARAMETER TUNING")
print("="*70)


svm_pipeline=Pipeline([
    ("scaler",StandardScaler()),
    ("classifier",
        SVC(
            probability=True,
            class_weight="balanced",
            random_state=42
        )
    )
])

svm_params={
    "classifier__C":[1,10,50],
    "classifier__kernel":["rbf"],
    "classifier__gamma":["scale",0.01,0.001]
}

svm_grid=GridSearchCV(
    svm_pipeline,
    svm_params,
    cv=cv,
    scoring="f1",
    n_jobs=-1
)

svm_grid.fit(X_train,y_train)

print("\nBest SVM Parameters")
print(svm_grid.best_params_)


rf_params={
    "n_estimators":[200,300,500],
    "max_depth":[10,20,None],
    "min_samples_split":[2,5],
    "class_weight":["balanced"]
}

rf_grid=GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_params,
    cv=cv,
    scoring="f1",
    n_jobs=-1
)

rf_grid.fit(X_train,y_train)

print("\nBest Random Forest Parameters")
print(rf_grid.best_params_)

# ---------------- XGBoost ----------------

if XGBOOST_AVAILABLE:

    scale_pos_weight=(
        y_train.value_counts()[0]/
        y_train.value_counts()[1]
    )

    xgb_params={
        "n_estimators":[200,300],
        "max_depth":[4,6],
        "learning_rate":[0.05,0.1],
        "subsample":[0.8,1.0],
        "colsample_bytree":[0.8,1.0]
    }

    xgb_grid=GridSearchCV(
        XGBClassifier(
            random_state=42,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight
        ),
        xgb_params,
        cv=cv,
        scoring="f1",
        n_jobs=-1
    )

    xgb_grid.fit(X_train,y_train)

    print("\nBest XGBoost Parameters")
    print(xgb_grid.best_params_)

# BEST MODELS

models={
    "SVM":svm_grid.best_estimator_,
    "Random Forest":rf_grid.best_estimator_
}

if XGBOOST_AVAILABLE:
    models["XGBoost"]=xgb_grid.best_estimator_

# TRAIN & EVALUATE

results=[]
predictions={}
probabilities={}
trained_models={}
best_thresholds={}

for name,model in models.items():

    print("\n"+"-"*70)
    print("Training:",name)
    print("-"*70)

    model.fit(X_train,y_train)

    y_prob=model.predict_proba(X_test)[:,1]

    precision,recall,thresholds=precision_recall_curve(
        y_test,
        y_prob
    )

    f1_scores=(
        2*(precision*recall)/
        (precision+recall+1e-10)
    )

    best_index=np.argmax(f1_scores[:-1])
    best_threshold=thresholds[best_index]

    y_pred=(y_prob>=best_threshold).astype(int)

    accuracy=accuracy_score(y_test,y_pred)
    prec=precision_score(y_test,y_pred)
    rec=recall_score(y_test,y_pred)
    f1=f1_score(y_test,y_pred)
    auc=roc_auc_score(y_test,y_prob)

    best_thresholds[name]=best_threshold

    results.append({
        "Model":name,
        "Threshold":round(best_threshold,3),
        "Accuracy":accuracy,
        "Precision":prec,
        "Recall":rec,
        "F1 Score":f1,
        "ROC-AUC":auc
    })

    predictions[name]=y_pred
    probabilities[name]=y_prob
    trained_models[name]=model

    print("Best Threshold:",round(best_threshold,3))
    print("Accuracy :",round(accuracy,4))
    print("Precision:",round(prec,4))
    print("Recall   :",round(rec,4))
    print("F1 Score :",round(f1,4))
    print("ROC-AUC  :",round(auc,4))

    print("\nClassification Report")
    print(classification_report(y_test,y_pred))

results_df=pd.DataFrame(results)

results_df=results_df.sort_values(
    by="F1 Score",
    ascending=False
)

print("\n"+"="*70)
print("MODEL COMPARISON")
print("="*70)

print(results_df.to_string(index=False))

results_df.to_csv("ML_Model_Comparison.csv",index=False)

metrics=[
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC"
]

for metric in metrics:

    plt.figure(figsize=(8,5))
    plt.bar(results_df["Model"],results_df[metric])
    plt.ylim(0,1)
    plt.title(metric+" Comparison")
    plt.tight_layout()
    plt.show()

# ============================================================
# r---rrr0e0dx ROC CURVE
# ============================================================

plt.figure(figsize=(8,6))

for name in models:

    fpr,tpr,_=roc_curve(y_test,probabilities[name])
    auc=roc_auc_score(y_test,probabilities[name])

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC={auc:.3f})"
    )

plt.plot([0,1],[0,1],"--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.show()

best_model_name=results_df.iloc[0]["Model"]
best_model=trained_models[best_model_name]

print("\nBest Model:",best_model_name)
#  CONFUSION MATRIX

cm=confusion_matrix(
    y_test,
    predictions[best_model_name]
)

print("\nConfusion Matrix")
print(cm)

ConfusionMatrixDisplay(cm).plot()
plt.title(best_model_name)
plt.show()
#  FEATURE IMPORTANCE

if best_model_name in ["Random Forest","XGBoost"]:

    importance=pd.DataFrame({
        "Feature":X.columns,
        "Importance":best_model.feature_importances_
    })

    importance=importance.sort_values(
        by="Importance",
        ascending=False
    )

    print("\nTop 20 Important Features")
    print(importance.head(20).to_string(index=False))

    importance.to_csv(
        "Feature_Importance.csv",
        index=False
    )

    plt.figure(figsize=(10,6))

    top=importance.head(15)

    plt.bar(top["Feature"],top["Importance"])

    plt.xticks(rotation=90)
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.show()
# FAILURE PROBABILITY

dataset["Failure_Probability"]=best_model.predict_proba(X)[:,1]
# PRIORITIZE TEST CASES

priority=dataset.sort_values(
    by="Failure_Probability",
    ascending=False
).copy()

priority.reset_index(drop=True,inplace=True)

priority["Priority"]=np.arange(len(priority))+1

print("\nTop 20 Prioritized Test Cases")

print(priority[
    [
        "Priority",
        "Test_ID",
        "Failure_Probability",
        "defects"
    ]
].head(20).to_string(index=False))

priority.to_csv(
    "Prioritized_Test_Cases.csv",
    index=False
)
#  PRIORITY GRAPH

top20=priority.head(20)

plt.figure(figsize=(12,5))

plt.bar(
    top20["Test_ID"],
    top20["Failure_Probability"]
)

plt.xticks(rotation=90)

plt.ylabel("Failure Probability")

plt.title("Top 20 Prioritized Test Cases")

plt.tight_layout()

plt.show()
#  APFD CALCULATION

fault_positions=np.where(
    priority["defects"].values==1
)[0]+1

m=len(fault_positions)
n=len(priority)

if m>0:

    apfd=1-(
        np.sum(fault_positions)/(m*n)
    )+1/(2*n)

    print("\nAPFD Score:",round(apfd,4))

#  FINAL SUMMARY

print("\n"+"="*70)
print("FINAL MODEL SUMMARY")
print("="*70)

print(results_df.to_string(index=False))

print("\nBEST MODEL:",best_model_name)
print("BEST THRESHOLD:",round(best_thresholds[best_model_name],3))
print("MODEL COMPARISON FILE: ML_Model_Comparison.csv")
print("PRIORITY FILE: Prioritized_Test_Cases.csv")

if best_model_name in ["Random Forest","XGBoost"]:
    print("FEATURE IMPORTANCE FILE: Feature_Importance.csv")

print("\nPROJECT COMPLETED SUCCESSFULLY")