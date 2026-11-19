"""Generate SHAP and LIME explanations for a fitted pipeline."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def explain(pipeline, X_train, X_test, output, n_background=60, n_explain=3):
    """Uses model-agnostic explanations on transformed features; potentially slow."""
    import shap
    from lime.lime_tabular import LimeTabularExplainer
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    pre=pipeline.named_steps['prep'];model=pipeline.named_steps['model']
    background=pre.transform(X_train.sample(min(n_background,len(X_train)),random_state=42))
    sample=pre.transform(X_test.iloc[:n_explain])
    feature_names=pre.get_feature_names_out().tolist()
    predict=lambda z: model.predict_proba(np.asarray(z))
    # Model-agnostic SHAP on a small sample; output can be costly on slower hardware.
    explainer=shap.Explainer(lambda z: predict(z)[:,1],background,algorithm='permutation')
    values=explainer(sample,max_evals=2*sample.shape[1]+1)
    shap.plots.waterfall(values[0],show=False,max_display=12)
    plt.tight_layout();plt.savefig(output/'shap_waterfall.png',dpi=150,bbox_inches='tight');plt.close()
    lime=LimeTabularExplainer(background,feature_names=feature_names,class_names=['No','Yes'],mode='classification',discretize_continuous=False,random_state=42)
    exp=lime.explain_instance(sample[0],predict,num_features=12,num_samples=600)
    exp.save_to_file(str(output/'lime_explanation.html'))
    print('Saved SHAP waterfall and LIME HTML to',output)

if __name__=='__main__':
    from train import ROOT,run
    pipe,xtr,xte,_,_=run(ROOT/'data/hospital_readmissions.csv',ROOT/'results')
    explain(pipe,xtr,xte,ROOT/'results/figures')
