import os
from pathlib import Path

import boto3
import pandas as pd
from moto import mock_aws
from pymongo import MongoClient

from src.constants import COLLECTION_NAME, DATABASE_NAME, MODEL_BUCKET_NAME, REGION_NAME
from src.pipline.prediction_pipeline import VehicleData, VehicleDataClassifier
from src.pipline.training_pipeline import TrainPipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "notebook" / "data.csv"
BATCH_SIZE = 5_000


def seed_mongodb() -> None:
    os.chdir(PROJECT_ROOT)
    dataset = pd.read_csv(DATASET_PATH)
    client = MongoClient(os.environ["MONGODB_URL"], serverSelectionTimeoutMS=10_000)
    try:
        client.admin.command("ping")
        collection = client[DATABASE_NAME][COLLECTION_NAME]
        existing_count = collection.count_documents({})

        if existing_count == 0:
            for start in range(0, len(dataset), BATCH_SIZE):
                records = dataset.iloc[start : start + BATCH_SIZE].to_dict("records")
                collection.insert_many(records)
        elif existing_count != len(dataset):
            raise RuntimeError(
                f"Expected an empty CI collection or {len(dataset)} records; "
                f"found {existing_count}."
            )

        print(f"MongoDB dataset ready: {len(dataset)} records.")
    finally:
        client.close()


def run_pipeline_and_prediction() -> None:
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = REGION_NAME

    with mock_aws():
        s3 = boto3.client("s3", region_name=REGION_NAME)
        s3.create_bucket(Bucket=MODEL_BUCKET_NAME)

        TrainPipeline().run_pipeline()

        vehicle = VehicleData(
            Gender=1,
            Age=44,
            Driving_License=1,
            Region_Code=28.0,
            Previously_Insured=0,
            Annual_Premium=40454.0,
            Policy_Sales_Channel=26.0,
            Vintage=217,
            Vehicle_Age_lt_1_Year=0,
            Vehicle_Age_gt_2_Years=1,
            Vehicle_Damage_Yes=1,
        )
        prediction = VehicleDataClassifier().predict(
            vehicle.get_vehicle_input_data_frame()
        )
        if int(prediction[0]) not in (0, 1):
            raise RuntimeError("Prediction smoke test returned an invalid class.")
        print(f"Prediction smoke test passed: class {int(prediction[0])}.")


if __name__ == "__main__":
    seed_mongodb()
    run_pipeline_and_prediction()
