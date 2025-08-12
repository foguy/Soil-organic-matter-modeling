from .utils import select_columns
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from mlens.ensemble import Subsemble
import numpy as np
np.int = int
import warnings

warnings.simplefilter('ignore')

class XgboostModel:
    def __init__(self, params = None):
        """
        Initialize the XgboostModel with optional parameters.
        
        :param params: Dictionary of parameters for the XGBRegressor.
        """
        if params is None:
            params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'reg:squarederror',
                'enable_categorical': True,
                'eval_metric': 'rmse'
            }
        self.model = XGBRegressor(**params)
        
    def fit(self, X, y, **kwargs):
        """
        Fit the model to the training data.
        
        :param X: Training features.
        :param y: Target variable.
        """
        self.model.fit(X, y, **kwargs)
        
    def predict(self, X):
        """
        Predict using the fitted model.
        
        :param X: Features for prediction.
        :return: Predicted values.
        """
        return self.model.predict(X)
    
    

class EnsembleModel:
    def __init__(self, 
                 params_1: dict = None, 
                 params_2: dict = None, 
                 params_3: dict = None, 
                 params_4: dict = None,
                 agri_prat_cols: list = None, 
                 phy_chi_cols: list = None, 
                 climatic_cols: list = None):
        """
        Initializes the EnsembleModel with base and meta models.
        """

        # Store parameter dicts
        self.params_1 = params_1 or {}
        self.params_2 = params_2 or {}
        self.params_3 = params_3 or {}
        self.params_4 = params_4 or {}

        # Store column groups
        self.agri_prat_cols = agri_prat_cols
        self.phy_chi_cols = phy_chi_cols
        self.climatic_cols = climatic_cols

        # Instantiate models
        self.base_model_1 = XGBRegressor(**self.params_1)
        self.base_model_2 = RandomForestRegressor(**self.params_2)
        self.base_model_3 = RandomForestRegressor(**self.params_3)
        self.meta_model = XGBRegressor(**self.params_4)

        # Build the ensemble pipeline
        self.build_ensemble_model()

    def build_ensemble_model(self):
        self.ensemble_model = Subsemble(partitions=5, shuffle=False, verbose=1)

        self.ensemble_model.add(
            estimators=[
                ('agri_prat_model', Pipeline([
                    ('select', select_columns(self.agri_prat_cols)),
                    ('model', self.base_model_1)
                ])),
                ('phy_chi_model', Pipeline([
                    ('select', select_columns(self.phy_chi_cols)),
                    ('model', self.base_model_2)
                ])),
                ('climatic_model', Pipeline([
                    ('select', select_columns(self.climatic_cols)),
                    ('model', self.base_model_3)
                ]))
            ],
            
        )

        self.ensemble_model.add_meta(self.meta_model)

    def fit(self, X, y):
        self.ensemble_model.fit(X, y)

    def predict(self, X):
        return self.ensemble_model.predict(X)
    
    