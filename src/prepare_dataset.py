from typing import Optional, List, Tuple, Union
import pandas as pd
from sklearn.model_selection import train_test_split
from .utils import fill_nan,  output_transform, extract_columns_types




class DataPreprocessor:
    def __init__(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        cat_cols:  Optional[List[str]] = None,
        num_cols:  Optional[List[str]] = None
    ):
        self.df               = df.copy()
        self.target_col       = target_col
        self.categorical_cols = cat_cols  or []
        self.numerical_cols   = num_cols  or []

    def _fill_missing(self):
        """ Fill NaNs using custom function. """
        self.df = fill_nan(self.df)

    def _infer_columns(self):
        
        """ Infer cat/num/target if the user didn’t supply them. """
        if not (self.categorical_cols and self.numerical_cols and self.target_col):
            cat_vars, num_vars, tgt_var = extract_columns_types(self.df)
            self.categorical_cols = self.categorical_cols or cat_vars
            self.numerical_cols   = self.numerical_cols   or num_vars
            self.target_col       = self.target_col       or tgt_var

    def _convert_dtypes(self):
        """ Bool → int, object → category. """
        bool_cols = self.df.select_dtypes(include='bool').columns
        if len(bool_cols):
            self.df[bool_cols] = self.df[bool_cols].astype(int)

        obj_cols = self.df.select_dtypes(include='object').columns
        for c in obj_cols:
            self.df[c] = self.df[c].astype('category')

    def preprocess(self) -> Tuple[pd.DataFrame, pd.Series]:
        
        
        self._fill_missing()
        
        self._infer_columns()
        
        self._convert_dtypes()

        
        X = self.df[self.categorical_cols + self.numerical_cols]
        y = self.df[self.target_col]
        return X, y, self.categorical_cols, self.numerical_cols, self.target_col


class DataModule:
    def __init__(
        self,
        csv_path: str,
        target_col: Optional[str]       = None,
        cat_cols:  Optional[List[str]] = None,
        num_cols:  Optional[List[str]] = None,
        train_size: float = 0.6,
        val_size:   float = 0.2,
        test_size:  float = 0.2,
        used_val:   bool   = False,
        transform_type: str = 'none',
        random_state:   int  = 42
    ):
        
        # require all part to be == 1
        if used_val:
            total = train_size + val_size + test_size
        else:
            total = train_size + test_size

        if abs(total - 1.0) > 1e-6:
            what = "train+val+test" if used_val else "train+test"
            raise ValueError(f"{what} must sum to 1 (got {total})")

        self.csv_path      = csv_path
        self.target_col    = target_col
        self.cat_cols      = cat_cols
        self.num_cols      = num_cols
        self.train_size    = train_size
        self.val_size      = val_size
        self.test_size     = test_size
        self.used_val      = used_val
        self.transform_type= transform_type
        self.random_state  = random_state

    def load(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Read the CSV, run fill/infer/convert,
        and return (features, target).
        """
        df = pd.read_csv(self.csv_path, index_col='scenario_id')
        pre = DataPreprocessor(
            df,
            target_col=self.target_col,
            cat_cols=self.cat_cols,
            num_cols=self.num_cols
        )
        X, y, cat_var, num_var, tgt_var = pre.preprocess()
        if self.transform_type not in ('none', None):
            y = output_transform(y, self.transform_type)
        return X, y, cat_var, num_var, tgt_var

    def split(
        self
    ) -> Union[
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]
    ]:
        """
        If used_val=False:
            returns X_train, y_train, X_test, y_test
        If used_val=True:
            returns X_train, y_train, X_val, y_val, X_test, y_test
        """
        X, y, _, _,_ = self.load()

        if not self.used_val:
            # Simple train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.random_state,
                shuffle=True
            )
            return X_train, y_train, X_test, y_test

        # Otherwise do train/val/test
        X_tv, X_test, y_tv, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=True
        )
        val_ratio = self.val_size / (self.train_size + self.val_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_tv, y_tv,
            test_size=val_ratio,
            random_state=self.random_state,
            shuffle=True
        )
        return X_train, y_train, X_val, y_val, X_test, y_test 