import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from src.FoodDelivery.components.feature_store import RedisFeatureStore
from src.FoodDelivery.loggers.logger import get_logger
from src.FoodDelivery.exceptions.exception import ExceptionsHandling
from config.config import *
import sys
import joblib
from pandas import DataFrame
from sklearn.impute import SimpleImputer

logger = get_logger(__name__)

class DataPreprocessing:
    def __init__(self, training_data, testing_data, feature_store:RedisFeatureStore):
        self.training_data = training_data
        self.testing_data = testing_data
        self.feature_store = feature_store
        self.train_data = None
        self.test_data = None
        self.top_features = [
            'multiple_deliveries', 'Road_traffic_density', 'Vehicle_condition', 'Delivery_person_Ratings',
            'distance_deliveries', 'Weather_conditions', 'Festival', 'distance_traffic', 'distance',
            'Delivery_person_Age', 'prep_traffic', 'City']
        
        logger.info("DataPreprocessing class initialized.")
        
        
    def load_data(self):
        try:
            self.train_data = pd.read_csv(TRAIN_PATH)
            self.test_data = pd.read_csv(TEST_PATH)
            logger.info("Data loaded successfully.")
            
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            raise ExceptionsHandling(e, sys) from e
        
        
        
    def  preprocessing_the_data(self, dataframe:DataFrame) ->DataFrame:
        """
        This function takes a DataFrame as input and performs the following preprocessing steps:
        1. Removes unwanted columns.
        2. Handles missing values.
        3. Encodes categorical variables.
        4. Scales numerical variables.
        5. Creates new features.
        6. Drops unnecessary columns.
        7. Returns the preprocessed DataFrame.
        9. Renames columns.
        """
        try:
            dataframe.rename(columns = {'Weatherconditions': 'Weather_conditions'}, inplace= True)
            dataframe.replace('Nan', float(np.nan), regex= True, inplace= True)
            if 'Time_taken(min)' in dataframe.columns:
                dataframe['Time_taken(min)'] = dataframe['Time_taken(min)'].apply(lambda x:int(x.split(' ')[1].strip()))
            dataframe['Weather_conditions'] = dataframe['Weather_conditions'].apply(lambda x: x.split(' ')[1].strip() if pd.notnull(x) else x)
            dataframe['City_code'] = dataframe['City_code'].str.split("RES", expand= True)[0]
            dataframe.drop(['Deliver_person_ID', 'ID'], inplace= True, axis = 1)
            dataframe['Delivery_person_Age'] = pd.to_numeric(dataframe['Delivery_person_Age'], errors='coerce').astype('float64')
            dataframe['Delivery_person_rating'] = pd.to_numeric(dataframe['Delivery_person_rating'], errors= 'coerce').astype('float64')
            dataframe['multiple_deliveries'] = pd.to_numeric(dataframe['multiple_deliveries'], errors= 'coerce').astype('float64')
            dataframe['Order_date'] = pd.to_datetime(dataframe['Order_date'], format= '%d-%m-%Y')
            dataframe['Weather_conditions'] = dataframe['Weather_conditions'].fillna(np.random.choice(dataframe['Weather_conditions'].dropna()))
            
            simple_imputer  = SimpleImputer(strategy= 'mode')
            dataframe['City'] = simple_imputer.fit_transform(dataframe[['City']])
            dataframe['Festival'] =  dataframe['Festival'].fillna(dataframe['Festival'].mode()[0])
            dataframe['multiple_deliveries'] = dataframe['multiple_deliveries'].fillna(dataframe['multiple_deliveries'].mode()[0])
            dataframe['Road_traffic_density'] = dataframe['Road_traffic_density'].fillna(dataframe['Road_traffic_density'].mode()[0])
            dataframe['Delivery_person_rating'] = dataframe['Delivery_person_rating'].fillna(dataframe['Delivery_person_rating'].median())
            
            
            logger.info(" If u have founded the error here than have a look in the processing or all done successfully")
            return dataframe
            
        except Exception as e:
            logger.info(f"Error in preprocessing data {e}")
            raise ExceptionsHandling(e, sys.exc_info())
        
    def encode_data(self, dataframe:DataFrame) ->DataFrame:
        """
        This function takes a DataFrame as input and performs the following encoding steps:
        1. Encodes categorical variables using Label Encoding.
        Retruns the encoded DataFrame.
        
        """
        try:
            categorical_features = dataframe.select_dtypes(include=['object']).columns
            label_encoder = LabelEncoder()
            for feature in categorical_features:
                dataframe[feature] = label_encoder.fit_transform(dataframe[feature])
            logger.info("Data encoded successfully.")
            return dataframe
        except Exception as e:
            logger.error(f"Error in encoding data: {e}")
            raise ExceptionsHandling(e, sys) from e
        
    
        