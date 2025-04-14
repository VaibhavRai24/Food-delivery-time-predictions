import os
RAW_DIR = "artifacts/raw"

TRAIN_PATH = os.path.join(RAW_DIR,'train.csv')
TEST_PATH = os.path.join(RAW_DIR,'test.csv')

PROCESSED_DIR = "artifacts/processed"

MODEL_DIR = "artifacts/models"

SCALER_PATH = os.path.join(MODEL_DIR,'scaler.pkl')

MODEL_PATH = os.path.join(MODEL_DIR,'model.pkl')