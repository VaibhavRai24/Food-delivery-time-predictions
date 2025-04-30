from src.FoodDelivery.components.ingestion import DataIngestion
from src.FoodDelivery.components.preprocessing import DataPreprocessing
from config.config import RAW_DIR, TRAIN_PATH, TEST_PATH
from src.FoodDelivery.loggers.logger import get_logger
import os


logger = get_logger(__name__)

class TrainingPipeline:
    def __init__(self):
        pass
    
    def DataIngestionStep(self):
        """
        This function is the main entry point for the data ingestion step.
        It creates an instance of the DataIngestion class and calls the run method.
        
        """
        try:
            logger.info("Starting data ingestion step.")
            data_ingestion = DataIngestion(output_dir= RAW_DIR)
            data_ingestion.run()
            
        except Exception as e:
            logger.error("Data ingestion process failed.")
            raise e
    
    
    def DataPreprocessingStep(self):
        """
        This function is the main entry point for the data preprocessing step.
        It creates an instance of the DataPreprocessing class and calls the run method.
        
        """
        try:
            logger.info("Starting data preprocessing step.")
            data_preprocessing = DataPreprocessing(training_data=TRAIN_PATH, testing_data =TEST_PATH)
            data_preprocessing.run()
            
        except Exception as e:
            logger.error("Data preprocessing process failed.")
            raise e
    
    def run_pipeline(self):
        """
        This function runs the entire pipeline by calling the DataIngestionStep function.
        
        """
        try:
            self.DataIngestionStep()
            self.DataPreprocessingStep()
            logger.info("Pipeline executed successfully.")
        except FileNotFoundError as e:
            logger.error("File not found. Please check the file paths.")
            raise e
        
    
    
