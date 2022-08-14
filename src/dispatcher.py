from sklearn import ensemble

MODLES =  {
    "randomforest": ensemble.RandomForestClassifier(n_estimators=200, n_jobs=-1,verbose=0),
    "extratrees": ensemble.ExtraTreesClassifier(n_estimators=200, n_jobs=-1,verbose=0)
}