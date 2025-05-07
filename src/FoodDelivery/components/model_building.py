import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score
import xgboost
from sklearn.preprocessing import StandardScaler
from src.FoodDelivery.loggers.logger import get_logger
from src.FoodDelivery.exceptions.exception import ExceptionsHandling
from config.config import MODEL_DIR, SCALER_PATH, MODEL_PATH
import os
import joblib
import sys
from pandas import DataFrame
from config.config import *

logger = get_logger(__name__)

class ModelBuilding:
    def __init__(self, train_path= TRAIN_PATH, test_path = TEST_PATH, model_saving_path=MODEL_PATH):
        self.train_path = train_path
        self.test_path = test_path
        self.model_saving_path = model_saving_path
        self.model = None
        self.scaler = StandardScaler()
        self.top_features = ['multiple_deliveries', 'Road_traffic_density', 'Vehicle_condition', 'Delivery_person_Ratings',
            'distance_deliveries', 'Weather_conditions', 'Festival', 'distance_traffic', 'distance',
            'Delivery_person_Age', 'prep_traffic', 'City'
            ]
        
        os.makedirs(self.model_saving_path, exist_ok=True)
        
        logger.info(f"Model saving path created at {self.model_saving_path}")
        logger.info("ModelBuilding class initialized")
        
        
    def load_data(self, file_path:str) ->DataFrame:
        try:
            if not os.path.exists(file_path):
                logger.info(f"File path {file_path} does not exist or found")
                return None
            data = pd.read_csv(file_path)
            return data
        except Exception as e:
            raise ExceptionsHandling(e, sys) from e
        
    def data_load_and_split(self):
        """
        This function is used to load the data from the csv file and split it into the training and testing data """
        try:
            logger.info(f"Loadinfg the training data from the {self.train_path}")
            train_df = self.load_data(self.train_path)
            
            logger.info(f"Loading the testing data from the {self.test_path}")
            test_df = self.load_data(self.test_path)
            
            train_df = pd.DataFrame(train_df)
            test_df = pd.DataFrame(test_df)
            
            X_train = train_df[self.top_features]
            X_test = test_df[self.top_features]
            y_train = train_df['Time_taken(min)']
            y_test = test_df['Time_taken(min)']
            
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            logger.info("Data loaded and split into training and testing data")
            return X_train_scaled, X_test_scaled, y_train, y_test
        except Exception as e:
            logger.error(f"Error in data loading and splitting in model building part: {e}")
            raise ExceptionsHandling(e, sys) from e
            
            
    def model_building_and_hyperparamter_tuning(self, X_train, y_train):
            try:
                param_grid_xgb = {
                'n_estimators': [50, 100, 150],
                'max_depth': [5, 7, 9],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0]
                }
                xgb_regressor = xgboost.XGBRegressor(random_state=42)
                random_search = RandomizedSearchCV(
                    xgb_regressor,
                    param_distributions=param_grid_xgb,
                    n_iter=10,
                    random_state=42,
                    n_jobs=-1,
                    scoring= 'r2',
                    cv=5
                )
                
                random_search.fit(X_train, y_train)
                logger.info("Model building and hyperparameter tuning completed")
                logger.info(f"Best parameters: {random_search.best_params_}")
                return random_search.best_estimator_
            except Exception as e:
                logger.error(f"Error during model building and hyperparameter tuning: {e}")
                raise ExceptionsHandling(e, sys) from e
            
    def model_evaluation(self, X_train, y_train, X_test, y_test):
            try:
                logger.info("Model Evaluation has been started")
                best_model = self.model_building_and_hyperparamter_tuning(X_train, y_train)
                best_model.fit(X_train, y_train)
                y_pred_train = best_model.predict(X_test)
                
                rmse = np.sqrt(mean_squared_error(y_test, y_pred_train))
                mae = mean_absolute_error(y_test, y_pred_train)
                r2 = r2_score(y_test, y_pred_train)
                auc = roc_auc_score(y_test, y_pred_train)
                
                logger.info(f"Model Evaluation completed with RMSE: {rmse}, MAE: {mae}, R2: {r2}, AUC: {auc}")
                self.save_model(best_model)
                self.save_scaler(self.scaler)
                return rmse, mae, r2, auc

            except Exception as e:
                logger.error(f"Error during model evaluation: {e}")
                raise ExceptionsHandling(e, sys) from e
            
    def save_model(self, model):
        try:
            model_path = MODEL_PATH
            joblib.dump(model, model_path)
            logger.info(f"Model saved at {model_path}")
        except Exception as e:
            logger.error(f"Error in saving the model: {e}")
            raise ExceptionsHandling(e, sys) from e
        
    def save_scaler(self, scaler):
        try:
            scaler_path = SCALER_PATH
            joblib.dump(scaler, scaler_path)
            logger.info(f"Scaler saved at {scaler_path}")
        except Exception as e:
            logger.error(f"Error in saving the scaler: {e}")
            raise ExceptionsHandling(e, sys) from e
        
    def run(self):
        try:
            logger.info("Model building process started")
            X_train, X_test, y_train, y_test = self.data_load_and_split()
            best_model , rmse, mae, r2, auc = self.model_evaluation(X_train, y_train, X_test, y_test)
            logger.info("Model building process completed")
            return best_model, rmse, mae, r2, auc
        except Exception as e:
            logger.error(f"Error in running the model building process: {e}")
            raise ExceptionsHandling(e, sys) from e
        
        
if __name__ == "__main__":
    train_path = TRAIN_PATH
    test_path = TEST_PATH
    model_saving_path = MODEL_DIR
    model_trainer = ModelBuilding(train_path, test_path, model_saving_path)
    model_trainer.run()
            
            
        
                    
            