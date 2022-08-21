

"""
- --- binary classification 
- --- multi class classification
- --- multi label classification
- --- single column regretion
- --- multi column regretion
- holdout
"""

from sklearn import model_selection
import pandas as pd


class CrossValidation:
    def __init__(
            self, 
            df, 
            target_cols,
            shuffle, 
            problem_type="binary_classification",
            multilabel_delimiter=",",
            num_folds=5,
            random_state=42,
        ):
        self.df = df
        self.target_cols = target_cols
        self.num_targets = len(target_cols)
        self.problem_type = problem_type
        self.num_folds = num_folds
        self.shuffle = shuffle
        self.random_state = random_state
        self.multilabel_delimiter=multilabel_delimiter

        if self.shuffle is True:
            self.df = self.df.sample(frac=1,random_state=self.random_state).reset_index(drop=True)
        
        self.df["kfold"] = -1

    def split(self):
        if self.problem_type in ("binary_classification", "multiclass_classification"):
            if self.num_targets !=1:
                raise Exception("Invalid number of targets for this problem type")
            target = self.target_cols[0]
            unique_values = self.df[target].nunique()
            if unique_values==1:
                raise Exception("Only one unique value found!")
            elif unique_values==2:
                kf = model_selection.StratifiedKFold(n_splits=self.num_folds, shuffle=False)
                for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.df, y=self.df[target].values)):
                    self.df.loc[val_idx, 'kfold'] = fold
        elif self.problem_type in ("single_col_regression", "multi_col_regression"):
            if self.num_targets !=1 and self.problem_type=="single_col_regression":
                raise Exception("Invalid number of targets for this problem type")
            if self.num_targets < 2 and self.problem_type=="multi_col_regression":
                raise Exception("Invalid number of targets for this problem type")
            target = self.target_cols[0]
            kf = model_selection.KFold(n_splits=5, shuffle=False)
            for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.df)):
                self.df.loc[val_idx, 'kfold'] = fold

        elif self.problem_type.startswith("holdout_"):
            # holdout_5, holdout_10, 
            holdout_persentage = int(self.problem_type.split("_")[1])
            num_holdout_samples = int(0.01*holdout_persentage*len(self.df))
            self.df.loc[:len(self.df)-num_holdout_samples,"kfold"] = 0
            self.df.loc[len(self.df)-num_holdout_samples:,"kfold"] = 1

        elif self.problem_type == "multilabel_classification":
            if self.num_targets !=1:
                raise Exception("Invalid number of targets for this problem type")
            targets =  self.df[self.target_cols[0]].apply(lambda x: len(str(x).split(self.multilabel_delimiter)))
            kf = model_selection.StratifiedKFold(n_splits=self.num_folds)
            for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.df, y=targets)):
                self.df.loc[val_idx, 'kfold'] = fold

        else:
            raise Exception("Proplem type not understood")

        return self.df

if __name__ == "__main__":
    df = pd.read_csv("input/train_multilabel.csv")
    cv = CrossValidation(df=df,shuffle=True,target_cols=['attribute_ids'],
                        problem_type="multilabel_classification", multilabel_delimiter=" ")
    df_split = cv.split()
    print(df_split.head())
    print(df_split.kfold.value_counts())
    df_split.to_csv("input/train_split.csv",index=False)

