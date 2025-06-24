import os
import pandas as pd
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import sys
from src.utils import save_object


from src.exception import CustomException
from src.logger import logging

import numpy as np

@dataclass
class DataTransformationConfig:
    preprocessed_data_path: str = os.path.join('artifacts', 'preprocessed.pkl')

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]

            numerical_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False))
            ])

            categorical_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False))  # Scaling for one-hot encoded features
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numerical_transformer, numerical_columns),
                    ("cat", categorical_transformer, categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            logging.error("Error occurred while creating data transformer object")
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Entered the data transformation method or component")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test dataframes")

            preprocessor = self.get_data_transformer_object()

            target_column = "math_score"
            X_train = train_df.drop(columns=[target_column], axis=1)
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column], axis=1)
            y_test = test_df[target_column]

            logging.info("Applying transformations on training and testing dataframes")

            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            logging.info("Transformations applied successfully")

            train_arr = np.c_[X_train_transformed, np.array(y_train)]
            test_arr = np.c_[X_test_transformed, np.array(y_test)]

            save_object(
                file_path= self.transformation_config.preprocessed_data_path,
                obj= preprocessor 
            )

            logging.info("Preprocessed data saved successfully")

            return (
                self.transformation_config.preprocessed_data_path,
                train_arr,
                test_arr
            )
           
        except Exception as e:
            logging.error("Error occurred during data transformation")
            raise CustomException(e, sys)