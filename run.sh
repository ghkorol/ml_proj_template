export TRAINING_DATA=input/train_folds.csv
export TEST_DATA=input/test_folds.csv
export MODEL=$1

echo "Fold 0 is starting:"
FOLD=0 python -m src.train
echo "Fold 1 is starting:"
FOLD=1 python -m src.train

