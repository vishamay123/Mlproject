import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor)
from sklearn.metrics import r2_score
# from sklearn.model_selection import GridSearchCV
# from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
# from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logger import logging  


from src.utils import save_object, evaluate_model


@dataclass
class ModelTrainingConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTraining:
    def __init__(self):
        self.model_training_config = ModelTrainingConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Entered the model training method or component")

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "XGBoost": XGBRegressor(),
                "KNeighbors": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "CatBoost": CatBoostRegressor(verbose=0)
            }

            model_report: dict = evaluate_model(X_train, y_train, X_test, y_test, models)

            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found with sufficient accuracy", sys)

            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")

            save_object(
                file_path=self.model_training_config.trained_model_file_path,
                obj=best_model
            )
            
            logging.info("Model training completed and model saved successfully")


            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            logging.info(f"R2 Score of the best model: {r2}")

            return r2

        except Exception as e:
            raise CustomException(e, sys)



# import os
# import sys
# from dataclasses import dataclass

# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor)
# from sklearn.metrics import r2_score
# from sklearn.model_selection import RandomizedSearchCV
# from xgboost import XGBRegressor
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.tree import DecisionTreeRegressor
# from catboost import CatBoostRegressor

# from src.exception import CustomException
# from src.logger import logging
# from src.utils import save_object


# @dataclass
# class ModelTrainingConfig:
#     trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')


# class ModelTraining:
#     def __init__(self):
#         self.model_training_config = ModelTrainingConfig()

#     def initiate_model_training(self, train_array, test_array):
#         try:
#             logging.info("Entered the model training method or component")

#             X_train, y_train = train_array[:, :-1], train_array[:, -1]
#             X_test, y_test = test_array[:, :-1], test_array[:, -1]

#             models = {
#                 "Random Forest": RandomForestRegressor(),
#                 "Gradient Boosting": GradientBoostingRegressor(),
#                 "AdaBoost": AdaBoostRegressor(),
#                 "XGBoost": XGBRegressor(),
#                 "KNeighbors": KNeighborsRegressor(),
#                 "Decision Tree": DecisionTreeRegressor(),
#                 "CatBoost": CatBoostRegressor(verbose=0)
#             }

#             params = {
#                 "Random Forest": {
#                     'n_estimators': [100, 200, 300],
#                     'max_depth': [None, 10, 20, 30],
#                     'min_samples_split': [2, 5, 10]
#                 },
#                 "Gradient Boosting": {
#                     'n_estimators': [100, 150, 200],
#                     'learning_rate': [0.01, 0.05, 0.1],
#                     'max_depth': [3, 5, 7]
#                 },
#                 "AdaBoost": {
#                     'n_estimators': [50, 100, 150],
#                     'learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0]
#                 },
#                 "XGBoost": {
#                     'n_estimators': [100, 200, 300],
#                     'learning_rate': [0.01, 0.05, 0.1],
#                     'max_depth': [3, 5, 7]
#                 },
#                 "KNeighbors": {
#                     'n_neighbors': [3, 5, 7, 9],
#                     'weights': ['uniform', 'distance'],
#                     'algorithm': ['auto', 'ball_tree', 'kd_tree']
#                 },
#                 "Decision Tree": {
#                     'max_depth': [None, 10, 20, 30],
#                     'min_samples_split': [2, 5, 10]
#                 },
#                 "CatBoost": {
#                     'iterations': [100, 200, 300],
#                     'learning_rate': [0.01, 0.05, 0.1],
#                     'depth': [3, 5, 7, 9]
#                 }
#             }

#             model_report = {}
#             best_models = {}

#             for model_name, model in models.items():
#                 logging.info(f"Tuning hyperparameters for {model_name}")
#                 param_grid = params.get(model_name, {})

#                 if param_grid:
#                     search = RandomizedSearchCV(
#                         model,
#                         param_distributions=param_grid,
#                         n_iter=10,
#                         cv=3,
#                         verbose=1,
#                         n_jobs=-1,
#                         random_state=42,
#                         scoring='r2'
#                     )
#                     search.fit(X_train, y_train)
#                     tuned_model = search.best_estimator_
#                 else:
#                     tuned_model = model
#                     tuned_model.fit(X_train, y_train)

#                 y_pred = tuned_model.predict(X_test)
#                 score = r2_score(y_test, y_pred)
#                 model_report[model_name] = score
#                 best_models[model_name] = tuned_model

#                 logging.info(f"{model_name} R2 Score: {score}")

#             best_model_score = max(model_report.values())
#             best_model_name = max(model_report, key=model_report.get)
#             best_model = best_models[best_model_name]

#             if best_model_score < 0.6:
#                 raise CustomException("No best model found with sufficient accuracy", sys)

#             logging.info(f"Best model: {best_model_name} with score: {best_model_score}")

#             save_object(
#                 file_path=self.model_training_config.trained_model_file_path,
#                 obj=best_model
#             )

#             logging.info("Model saved successfully")

#             final_y_pred = best_model.predict(X_test)
#             r2 = r2_score(y_test, final_y_pred)

#             return r2

#         except Exception as e:
#             raise CustomException(e, sys)
