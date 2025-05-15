import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from src.FoodDelivery.loggers.logger import get_logger
from src.FoodDelivery.exceptions.exception import ExceptionsHandling
from config.config import *
import sys
import joblib
from geopy.distance import geodesic
from pandas import DataFrame
from sklearn.impute import SimpleImputer

logger = get_logger(__name__)

class DataPreprocessing:
    def __init__(self, training_data, testing_data, feature_store= None):
        self.training_data = training_data
        self.testing_data = testing_data
        self.feature_store = feature_store
        self.train_data = None
        self.test_data = None
        self.data = None
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
        
    def save_data(self, df:pd.DataFrame, file_path:str) -> None:
        """
        We are making out this fun to save the data in the form of csv at the given file path
        
        """
        try:
            df.to_csv(file_path, index= False)
            logger.info(f" Saved the file at the given place successfully{file_path}")
            
        except Exception as e:
            logger.info(" Some error has taken place while saving the file")
            raise ExceptionsHandling(e, sys)   
        
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
            dataframe['City_code'] = dataframe['Delivery_person_ID'].str.split("RES", expand= True)[0]
            dataframe.drop(['Delivery_person_ID', 'ID'], inplace= True, axis = 1)
            dataframe['Delivery_person_Age'] = pd.to_numeric(dataframe['Delivery_person_Age'], errors='coerce').astype('float64')
            dataframe['Delivery_person_Ratings'] = pd.to_numeric(dataframe['Delivery_person_Ratings'], errors= 'coerce').astype('float64')
            dataframe['multiple_deliveries'] = pd.to_numeric(dataframe['multiple_deliveries'], errors= 'coerce').astype('float64')
            dataframe['Order_Date'] = pd.to_datetime(dataframe['Order_Date'], format= '%d-%m-%Y')
            dataframe['Weather_conditions'] = dataframe['Weather_conditions'].fillna(np.random.choice(dataframe['Weather_conditions'].dropna()))
            
            dataframe['City'] = dataframe['City'].fillna(dataframe['City'].mode()[0])
            dataframe['Festival'] =  dataframe['Festival'].fillna(dataframe['Festival'].mode()[0])
            dataframe['multiple_deliveries'] = dataframe['multiple_deliveries'].fillna(dataframe['multiple_deliveries'].mode()[0])
            dataframe['Road_traffic_density'] = dataframe['Road_traffic_density'].fillna(dataframe['Road_traffic_density'].mode()[0])
            dataframe['Delivery_person_Ratings'] = dataframe['Delivery_person_Ratings'].fillna(dataframe['Delivery_person_Ratings'].median())
            
            
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
        
    def feature_engineering(self, dataframe: DataFrame) -> DataFrame:
        
        try:
            dataframe["day"] = dataframe["Order_Date"].dt.day
            dataframe["month"] = dataframe["Order_Date"].dt.month
            dataframe["year"] = dataframe["Order_Date"].dt.year
            dataframe["quarter"] = dataframe["Order_Date"].dt.quarter
            dataframe["day_of_week"] = dataframe["Order_Date"].dt.dayofweek.astype("int")
            dataframe["is_month_end"] = dataframe["Order_Date"].dt.is_month_end.astype("int")
            dataframe["is_month_start"] = dataframe["Order_Date"].dt.is_month_start.astype("int")
            dataframe["is_quarter_end"] = dataframe["Order_Date"].dt.is_quarter_end.astype("int")
            dataframe["is_quarter_start"] = dataframe["Order_Date"].dt.is_quarter_start.astype("int")
            dataframe["is_year_end"] = dataframe["Order_Date"].dt.is_year_end.astype("int")
            dataframe["is_year_start"] = dataframe["Order_Date"].dt.is_year_start.astype("int")
            dataframe["is_weekend"] = dataframe["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)
            dataframe["is_weekday"] = dataframe["day_of_week"].apply(lambda x: 1 if x < 5 else 0)
            dataframe["is_holiday"] = dataframe["Festival"].apply(lambda x: 1 if x == "Yes" else 0)
            dataframe['Time_Orderd'] = dataframe['Time_Orderd'].str.strip()
            dataframe["Time_Orderd"] = pd.to_timedelta(dataframe["Time_Orderd"].fillna("00:00:00"), errors='coerce')
            dataframe['Time_Order_picked'] = dataframe['Time_Order_picked'].str.strip()
            dataframe["Time_Order_picked"] = pd.to_timedelta(dataframe["Time_Order_picked"].fillna("00:00:00"), errors='coerce')
            dataframe["Time_Ordered_formattted"] = dataframe['Order_Date'] + dataframe["Time_Orderd"]
            dataframe["Time_Order_picked_base"] = dataframe['Order_Date'] + dataframe["Time_Order_picked"]
            mask = dataframe['Time_Order_picked'] < dataframe['Time_Orderd']
            dataframe.loc[mask, 'Time_Order_picked'] = dataframe.loc[mask, 'Time_Order_picked'] + pd.Timedelta(days=1)
            dataframe["Order_prepare_time"] = (dataframe["Time_Ordered_formattted"] - dataframe["Time_Ordered_formattted"]).dt.total_seconds() / 60
            dataframe["Order_prepare_time"] = dataframe["Order_prepare_time"].fillna(dataframe["Order_prepare_time"].median())
            
            dataframe.drop(["Time_Orderd", "Time_Order_picked", "Order_Date", "Time_Ordered_formattted", "Time_Order_picked_base"], axis=1, inplace=True)
            restraunt_coordinates = dataframe[["Delivery_location_latitude", "Delivery_location_longitude"]].to_numpy()
            delivery_location_coordinates  = dataframe[["Delivery_location_latitude", "Delivery_location_longitude"]].to_numpy()
            dataframe["distance"] = np.array([geodesic(restraunt, delivery).kilometers for restraunt, delivery in zip(restraunt_coordinates, delivery_location_coordinates)])
            dataframe  =self.encode_data(dataframe)
            dataframe['distance_traffic'] = dataframe['distance'] * dataframe['Road_traffic_density']
            dataframe['distance_deliveries'] = dataframe['distance'] * dataframe['multiple_deliveries']
            dataframe['prep_traffic'] = dataframe['Order_prepare_time'] * dataframe['Road_traffic_density']
            dataframe['age_ratings'] = dataframe['Delivery_person_Age'] * dataframe['Delivery_person_Ratings']
            dataframe['prep_distance'] = dataframe['Order_prepare_time'] * dataframe['distance']
            
            for col in ['distance', 'Order_prepare_time', 'Delivery_person_Age', 'multiple_deliveries']:
                upper_limit = dataframe[col].quantile(0.99)
                dataframe[col] = np.where(dataframe[col] > upper_limit, upper_limit, dataframe[col])
                
            logger.info("Feature engineering completed successfully.")
            return dataframe
        
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            raise ExceptionsHandling(e, sys) from e
        
        
    def split_data(self, dataframe:DataFrame):
        try:
            X = dataframe.drop(columns=['Time_taken(min)'])
            y = dataframe['Time_taken(min)']
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.X_train = self.X_train[self.top_features]
            self.X_test = self.X_test[self.top_features]
            self.test_data = self.test_data[self.top_features]
            
            scaler = StandardScaler()
            self.X_train = pd.DataFrame(scaler.fit_transform(self.X_train), columns=self.top_features)
            self.X_test = pd.DataFrame(scaler.transform(self.X_test), columns=self.top_features)
            self.test_data = pd.DataFrame(scaler.transform(self.test_data), columns=self.top_features)

            
            os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
            joblib.dump(scaler, SCALER_PATH)
            os.makedirs(os.path.dirname(PROCESSED_TRAIN_DATA_PATH), exist_ok=True)
            os.makedirs(os.path.dirname(PROCESSED_TEST_DATA_PATH), exist_ok=True)
            
            self.save_data(pd.DataFrame(self.X_train), PROCESSED_TRAIN_DATA_PATH)
            self.save_data(pd.DataFrame(self.X_test), PROCESSED_TEST_DATA_PATH)
            self.save_data(pd.DataFrame(self.test_data), PROCESSED_TESTING_DATA_PATH)
    
            os.makedirs(os.path.dirname(PROCESSED_TRAIN_TARGET_PATH), exist_ok=True)
            os.makedirs(os.path.dirname(PROCESSED_TEST_TARGET_PATH), exist_ok=True)

            self.save_data(pd.DataFrame(self.y_train, columns=['Time_taken(min)']), PROCESSED_TRAIN_TARGET_PATH)
            self.save_data(pd.DataFrame(self.y_test, columns=['Time_taken(min)']), PROCESSED_TEST_TARGET_PATH)

            logger.info("Data split and scaling completed successfully.")
            
        except Exception as e:
            logger.error(f"Error in splitting data: {e}")
            raise ExceptionsHandling(e, sys) from e
    
    
    def run(self):
        try:
            logger.info("Starting data preprocessing...")
            self.load_data()
            
            self.training_data =  self.train_data.copy()
            self.training_data  = self.preprocessing_the_data(self.training_data)
            self.training_data = self.feature_engineering(self.training_data)
            self.training_data = self.encode_data(self.training_data)
            
            self.test_data = self.test_data.copy()
            self.test_data = self.preprocessing_the_data(self.test_data)
            self.test_data = self.feature_engineering(self.test_data)
            self.test_data = self.encode_data(self.test_data)
            
            self.split_data(self.training_data)
            
            logger.info("Data preprocessing completed successfully.")
        except Exception as e:
            logger.error(f"Error in data preprocessing: {e}")
            raise ExceptionsHandling(e, sys) from e
        
if __name__ == "__main__":
    data_preprocessing = DataPreprocessing(training_data=None, testing_data=None)
    data_preprocessing.run()
    logger.info("Data preprocessing script executed successfully.")