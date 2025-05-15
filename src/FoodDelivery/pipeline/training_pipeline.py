from src.FoodDelivery.components.ingestion import DataIngestion
from src.FoodDelivery.components.preprocessing import DataPreprocessing
from src.FoodDelivery.components.model_building import ModelBuilding
from config.config import RAW_DIR, TRAIN_PATH, TEST_PATH, PROCESSED_TRAIN_DATA_PATH, PROCESSED_TEST_DATA_PATH, MODEL_PATH
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
        
        
    def modelBuiding(self):
        """
        This is the function which is used to run the model building part 
        
        """
        
        try:
            logger.info("Stating the model building part")
            model_building = ModelBuilding(train_path=PROCESSED_TRAIN_DATA_PATH, test_path= PROCESSED_TEST_DATA_PATH, model_saving_path= MODEL_PATH)
            model_building.run()
            
        except Exception as e:
            logger.error("File not found or error in the some of the function in the model building parrt.......")
            raise e
    
    def run_pipeline(self):
        """
        This function runs the entire pipeline by calling the DataIngestionStep function.
        
        """
        try:
            self.DataIngestionStep()
            self.DataPreprocessingStep()
            self.modelBuiding()
            logger.info("Pipeline executed successfully.")
        except FileNotFoundError as e:
            logger.error("File not found. Please check the file paths.")
            raise e
        
    
    
