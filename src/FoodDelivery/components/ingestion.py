import sys
import os
import pandas as pd
from src.FoodDelivery.exceptions.exception import ExceptionsHandling
from src.FoodDelivery.logger.logger import get_logger
import kaggle
from config.config import RAW_DIR, TRAIN_PATH, TEST_PATH

logger = get_logger(__name__)


class DataIngestion:
    """
    This class is responsible for downloading and extracting the dataset from Kaggle.
    It uses the Kaggle API to download the dataset and extract it to a specified directory.
    
    """
    def __init__(self, output_dir:str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok= True)
        
        try:
            kaggle.api.authenticate()
            logger.info("Kaggle api authentication successful.")
        except Exception as e:
            logger.error("Kaggle api authentication failed.")
            raise ExceptionsHandling(e, sys) from e
        
        
    def download_dataset(self):
        try:
            dataset = "gauravmalik26/food-delivery-dataset"
            kaggle.api.dataset_download_files(dataset, path= self.output_dir, unzip= True)
            logger.info("Dataset downloaded successfully.")
            
            
            train_file = None
            test_file = None
            
            for file in os.listdir(self.output_dir):
                if file == 'train.csv':
                    train_file = os.path.join(self.output_dir, file)
                elif file == 'test.csv':
                    test_file = os.path.join(self.output_dir, file)
                elif file not in ['train.csv', 'test.csv']:
                    os.remove(os.path.join(self.output_dir, file))
                    logger.info(f"Removed unwanted file: {file}")
                    
            if not train_file or not test_file:
                raise FileNotFoundError("train.csv or test.csv not found in the downloaded files.")
            
            
            train_df  = pd.read_csv(train_file)
            test_df = pd.read_csv(test_file)
            logger.info(f"Loaded the train file with {len(train_df)} and test file with {len(test_df)} records.")
            
            return train_df, test_df
        
            
        except Exception as e:
            logger.error("Dataset download failed.")
            raise ExceptionsHandling(e, sys) from e
        
    
    def save_data(self, train_df, test_df):
        try:
            logger.info("Saving train and test data to the specified directory.")
            train_df.to_csv(TRAIN_PATH, index= False)
            test_df.to_csv(TEST_PATH, index= False)
            logger.info(f"Data saved successfully at the {TRAIN_PATH} and {TEST_PATH}.")
            
        except Exception as e:
            logger.error("Failed to save the data.")
            raise ExceptionsHandling(e, sys) from e
        
    def run(self):
        """
        This method runs the data ingestion process.
        It downloads the dataset and saves it to the specified directory.
        """
        try:
            logger.info("Starting data ingestion process.")
            train_df, test_df = self.download_dataset()
            self.save_data(train_df, test_df)
            logger.info("Data ingestion process completed successfully.")
            
        except Exception as e:
            logger.error("Data ingestion process failed.")
            raise ExceptionsHandling(e, sys) from e
        

if __name__ == "__main__":
    try:
        output_dir = RAW_DIR
        data_ingestion = DataIngestion(output_dir)
        data_ingestion.run()
        
    except Exception as e:
        logger.error("Data ingestion process failed.")
        raise ExceptionsHandling(e, sys) from e