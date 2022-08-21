

"""
---- binary classification 
---- multi class classification
- multi label classification
- single column regretion
- multi column regretion
- holdout
"""

from sklearn import model_selection
import pandas as pd


class CrossValidation:
    def __init__(
            self, 
            df, 
            target_cols, 
            problem_type="binary_classification",
            num_folds=5,
            shuffle=True,
            random_state=42,
        ):
        self.df = df
        self.target_cols = target_cols
        self.num_targets = len(target_cols)
        self.problem_type = problem_type
        self.num_folds = num_folds
        self.shuffle = shuffle
        self.random_state = random_state

        if self.shuffle is True:
            self.df = self.df.sample(frac=1,random_state=self.random_state).reset_index(drop=True)
        
        self.df["kfold"] = -1

    def split(self):
        if self.problem_type in ["binary_classification", "multiclass_classification"]:
            if self.num_targets !=1:
                raise Exception("Invalid number of targets for this problem type")
            target = self.target_cols[0]
            unique_values = self.df[target].nunique()
            if unique_values==1:
                raise Exception("Only one unique value found!")
            elif unique_values==2:
                kf = model_selection.StratifiedKFold(n_splits=5, shuffle=False)

                for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.df, y=self.df[target].values)):
                    self.df.loc[val_idx, 'kfold'] = fold
        elif self.problem_type == "single_col_regression":
            if self.num_targets !=1:
                raise Exception("Invalid number of targets for this problem type")
            target = self.target_cols[0]
            kf = model_selection.KFold(n_splits=5, shuffle=False)
            for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.df, y=self.df[target].values)):
                self.df.loc[val_idx, 'kfold'] = fold

        else:
            raise Exception("Proplem type not understood")

        return self.df

if __name__ == "__main__":
    df = pd.read_csv("input/train_reg.csv")
    cv = CrossValidation(df=df,target_cols=['SalePrice'],problem_type="single_col_regression")
    df_split = cv.split()
    print(df_split.head())
    print(df_split.kfold.value_counts())
    df_split.to_csv("input/train_split.csv",index=False)

