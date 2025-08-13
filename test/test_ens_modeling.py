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



phy_chi_cols = [ 'latitude', 'sand_perc', 'clay_perc', 'bulk_density', 'ph',
                'soc', 'drain_depth', 'drain_spacing']
climatic_cols = ['precipitation_cum_annual_1', 'precipitation_cum_annual_2',
                'precipitation_cum_annual_3', 'precipitation_cum_annual_4',
                'precipitation_cum_annual_5',
                'precipitation_shannon_diversity_index_annual_1',
                'precipitation_shannon_diversity_index_annual_2',
                'precipitation_shannon_diversity_index_annual_3',
                'precipitation_shannon_diversity_index_annual_4',
                'precipitation_shannon_diversity_index_annual_5',
                'tdd_cum_annual_1', 'tdd_cum_annual_2',
                'tdd_cum_annual_3', 'tdd_cum_annual_4',
                'tdd_cum_annual_5']
agri_prat_cols = ['apps_mineral_1', 'apps_mineral_2', 'apps_mineral_3',
                'apps_mineral_4', 'apps_mineral_5', 'apps_organic_1',
                'apps_organic_2', 'apps_organic_3', 'apps_organic_4',
                'apps_organic_5', 'catch_crop_1', 'catch_crop_2',
                'catch_crop_3', 'catch_crop_4', 'catch_crop_5',
                'inter_crop_1', 'inter_crop_2', 'inter_crop_3',
                'inter_crop_4', 'inter_crop_5', 'is_residues_left',
                'is_tillage', 'main_crop_1', 'main_crop_2', 'main_crop_3',
                'main_crop_4', 'main_crop_5']


params_1 = {'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'reg:squarederror',
                'enable_categorical': True,
                'eval_metric': 'rmse'
            }

params_2 = {
    "n_estimators": 500,
    "max_depth": 4,
    "random_state": 42,
}

params_3 = {
    "n_estimators": 500,
    "max_depth": 4,

    "random_state": 42,
}

params_4 = {
    "n_estimators": 500,
    "max_depth": 4,

    "random_state": 42,
}


ens = EnsembleModel(params_1= params_1, 
                    params_2 = params_2, 
                    params_3 =params_3, 
                    params_4 = params_4, 
                    phy_chi_cols = phy_chi_cols, 
                    climatic_cols = climatic_cols, 
                    agri_prat_cols =agri_prat_cols)
ens.fit(X_train, y_train)


y_pred = ens.predict(X_test)


evaluation = EvaluateTask(y_true=y_test, y_pred=y_pred)

evaluation_results = evaluation.evaluate_model()

print(type(ens))

for k,v in evaluation_results.items():
    print(f"{k}: {v:.4f}")
    