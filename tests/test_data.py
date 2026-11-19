from src.train import load_data,ROOT

def test_dataset():
    X,y=load_data(ROOT/'data/hospital_readmissions.csv')
    assert len(X)==len(y)==25000
    assert set(y.unique())=={0,1}
    assert 'readmitted' not in X
