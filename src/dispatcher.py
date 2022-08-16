from sklearn import ensemble

#Score: 0.74404
#Public score: 0.75077
MODLES =  {
    "randomforest": ensemble.RandomForestClassifier(n_estimators=200, n_jobs=-1,verbose=0),
    "extratrees": ensemble.ExtraTreesClassifier(n_estimators=200, n_jobs=-1,verbose=0)
}