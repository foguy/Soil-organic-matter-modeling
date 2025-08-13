import sys
import os.path as path
two_up =  path.abspath(path.join(__file__ ,"../.."))
sys.path.append(two_up)

from src.ml_models import XgboostModel, EnsembleModel
from src.prepare_dataset import DataPreprocessor, DataModule 
from src.evaluation import EvaluateTask
from src.utils import plot_feature_importance


csv_path = '/home/fotg2901/project/Som_prediction/Soil-organic-matter-modeling/data/final_dataset_scenario.csv'


data_module = DataModule(csv_path=csv_path, train_size=0.8, test_size=0.2,  used_val = False)
df, target, cat_var, num_var, tgt_var = data_module.load()

print(f"categorical variables :\n{cat_var}")
print(f"numerical variables:\n{num_var}")
print(f"target variable:\n{tgt_var}")

X_train, y_train, X_test, y_test = data_module.split()


print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

estimat = XgboostModel()
estimat.fit(X_train, y_train)

y_pred = estimat.predict(X_test)

print(y_pred)

evaluation = EvaluateTask(y_true=y_test, y_pred=y_pred)

evaluation_results = evaluation.evaluate_model()

for k,v in evaluation_results.items():
    print(f"{k}: {v:.4f}")
    
    
plot_feature_importance(estimat.model, X_train, top_n=10, show=True)
