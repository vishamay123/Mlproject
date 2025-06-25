import os
import sys


import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import dill
from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
   
   
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)
            
      
    except Exception as e:
        logging.error(f"Error occurred while saving object: {e}")
        raise CustomException(e, sys)
    

def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)
            report[model_name] = score
            logging.info(f"{model_name} R2 Score: {score}")
        
        return report
    
    except Exception as e:
        logging.error(f"Error occurred during model evaluation: {e}")
        raise CustomException(e, sys)
    
def load_object(file_path): 
    try:
        with open(file_path, 'rb') as file:
            return dill.load(file)
    except Exception as e:
        logging.error(f"Error occurred while loading object: {e}")
        raise CustomException(e, sys)