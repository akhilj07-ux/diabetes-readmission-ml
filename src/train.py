"""Reconstructed diabetes readmission benchmark; no patient-level claims."""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

class CompatibleCatBoost(ClassifierMixin, BaseEstimator):
    def __init__(self, iterations=180, depth=5, learning_rate=0.05, random_seed=42):
        self.iterations=iterations;self.depth=depth;self.learning_rate=learning_rate;self.random_seed=random_seed
    def fit(self,X,y):
        self.model_=CatBoostClassifier(iterations=self.iterations,depth=self.depth,learning_rate=self.learning_rate,random_seed=self.random_seed,verbose=False,thread_count=2)
        self.model_.fit(X,y);self.classes_=np.array([0,1]);return self
    def predict(self,X):return self.model_.predict(X).reshape(-1).astype(int)
    def predict_proba(self,X):return self.model_.predict_proba(X)


ROOT=Path(__file__).resolve().parents[1]

def load_data(path):
    df=pd.read_csv(path)
    if 'readmitted' not in df: raise ValueError('Missing readmitted target')
    y=df.pop('readmitted').astype(str).str.strip().str.lower().map({'yes':1,'no':0})
    if y.isna().any(): raise ValueError('Unexpected target label')
    # Encounter history features are available in this supplied dataset; prospective availability must be assessed separately.
    return df,y

def make_preprocessor(X):
    numeric=X.select_dtypes(include='number').columns.tolist()
    categorical=[c for c in X.columns if c not in numeric]
    return ColumnTransformer([
        ('numeric',Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler())]),numeric),
        ('categorical',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),categorical)
    ],sparse_threshold=0)

def models(seed):
    return {
      'Logistic Regression':LogisticRegression(max_iter=1200,random_state=seed),
      'Decision Tree':DecisionTreeClassifier(max_depth=7,min_samples_leaf=30,random_state=seed),
      'Random Forest':RandomForestClassifier(n_estimators=120,min_samples_leaf=5,n_jobs=-1,random_state=seed),
      'SVM':CalibratedClassifierCV(LinearSVC(C=0.1,dual=False,random_state=seed,max_iter=2000),cv=3),
      'Gradient Boosting':HistGradientBoostingClassifier(max_iter=120,random_state=seed),
      'CatBoost':CompatibleCatBoost(iterations=180,depth=5,learning_rate=0.05,random_seed=seed),
      'MLP':MLPClassifier(hidden_layer_sizes=(48,24),max_iter=120,early_stopping=True,random_state=seed)
    }

def run(data,output,quick=False,cv=False):
    output.mkdir(parents=True,exist_ok=True)
    X,y=load_data(data)
    Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
    results=[]; trained={}
    for name,model in models(42).items():
        print('Training:',name,flush=True)
        pipeline=Pipeline([('prep',make_preprocessor(Xtrain)),('model',model)])
        if cv:
            scores=cross_validate(pipeline,Xtrain,ytrain,cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=42),scoring={'roc_auc':'roc_auc','f1':'f1'},n_jobs=1)
            cv_auc=float(np.mean(scores['test_roc_auc']))
        else: cv_auc=np.nan
        pipeline.fit(Xtrain,ytrain)
        prob=pipeline.predict_proba(Xtest)[:,1]
        pred=(prob>=.5).astype(int)
        results.append({'model':name,'accuracy':accuracy_score(ytest,pred),'precision':precision_score(ytest,pred,zero_division=0),'recall':recall_score(ytest,pred,zero_division=0),'f1':f1_score(ytest,pred,zero_division=0),'roc_auc':roc_auc_score(ytest,prob),'average_precision':average_precision_score(ytest,prob),'cv_roc_auc':cv_auc})
        trained[name]=pipeline
    table=pd.DataFrame(results).sort_values('roc_auc',ascending=False)
    table.to_csv(output/'model_comparison.csv',index=False)
    best_name=table.iloc[0]['model'];best=trained[best_name]
    ConfusionMatrixDisplay.from_estimator(best,Xtest,ytest,values_format='d');plt.title('Holdout confusion matrix: '+best_name);plt.tight_layout();plt.savefig(output/'figures/confusion_matrix.png',dpi=150);plt.close()
    RocCurveDisplay.from_estimator(best,Xtest,ytest);plt.title('Holdout ROC: '+best_name);plt.tight_layout();plt.savefig(output/'figures/roc_curve.png',dpi=150);plt.close()
    table.sort_values('roc_auc').plot.barh(x='model',y='roc_auc',legend=False,figsize=(8,4));plt.xlabel('Holdout ROC AUC');plt.tight_layout();plt.savefig(output/'figures/model_comparison.png',dpi=150);plt.close()
    metadata={'rows':len(X),'features':len(X.columns),'train_rows':len(Xtrain),'test_rows':len(Xtest),'seed':42,'best_model_by_holdout_auc':best_name,'note':'Exploratory benchmark only. Selecting best model on test set biases the reported best-model estimate. No clinical validation.'}
    (output/'run_metadata.json').write_text(json.dumps(metadata,indent=2))
    print(table.to_string(index=False));print('Best:',best_name)
    return best,Xtrain,Xtest,ytrain,ytest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=ROOT/'data/hospital_readmissions.csv');p.add_argument('--output',type=Path,default=ROOT/'results');p.add_argument('--cv',action='store_true');a=p.parse_args()
    run(a.data,a.output,cv=a.cv)
