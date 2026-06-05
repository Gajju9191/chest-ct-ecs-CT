#!/usr/bin/env python3
"""
AWS Batch Retraining Script with DAGsHub MLflow Tracking
"""

import boto3
import tensorflow as tf
import mlflow
import mlflow.tensorflow
import os
import requests
import json
import logging
import numpy as np
import zipfile
from datetime import datetime
from pathlib import Path
from requests.auth import HTTPBasicAuth

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration - Get from environment variables
MODEL_BUCKET = os.environ.get('MODEL_BUCKET', 'chest-ct-models-155407238004')
DATA_BUCKET = os.environ.get('DATA_BUCKET', 'chest-models-gajju')
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')
JENKINS_URL = os.environ.get('JENKINS_URL', 'http://52.91.160.186:8080')
JENKINS_TOKEN = os.environ.get('JENKINS_TOKEN', 'ct-trigger-token')
JENKINS_USERNAME = os.environ.get('JENKINS_USERNAME', 'Gajanan Wagalgave')
JENKINS_API_TOKEN = os.environ.get('JENKINS_API_TOKEN', '118d61c306e6cdc524e373e47263b305b1')
JOB_NAME = "first-chest-pipeline"

# MLflow Remote Tracking Configuration (DAGsHub)
MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI', 'https://dagshub.com/Gajju9191/chest-ct-ecs.mlflow')
MLFLOW_TRACKING_USERNAME = os.environ.get('MLFLOW_TRACKING_USERNAME', 'Gajju9191')
MLFLOW_TRACKING_PASSWORD = os.environ.get('MLFLOW_TRACKING_PASSWORD', '089e1f4ec33ad67cc8541160fe89a199ce77186d')

# Set MLflow tracking URI
if MLFLOW_TRACKING_PASSWORD:
    os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
    os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info(f"✅ MLflow tracking configured: {MLFLOW_TRACKING_URI}")
else:
    logger.warning("⚠️ MLFLOW_TRACKING_PASSWORD not set. Experiments will be logged locally.")

# Local paths
CURRENT_MODEL_PATH = '/tmp/current_model.h5'
NEW_MODEL_PATH = '/tmp/new_model.h5'
DATA_PATH = '/tmp/data/'
BATCH_SIZE = 32
FINE_TUNE_EPOCHS = 10
LEARNING_RATE = 1e-5


def trigger_jenkins():
    """Trigger Jenkins deployment with CSRF crumb handling"""
    try:
        # First, get the CSRF crumb
        crumb_url = f"{JENKINS_URL}/crumbIssuer/api/json"
        crumb_resp = requests.get(crumb_url, timeout=10)
        
        headers = {}
        if crumb_resp.status_code == 200:
            crumb_data = crumb_resp.json()
            crumb = crumb_data['crumb']
            crumb_field = crumb_data['crumbRequestField']
            headers = {crumb_field: crumb}
            logger.info("✅ CSRF crumb obtained")
        else:
            logger.warning(f"Could not obtain CSRF crumb: {crumb_resp.status_code}")
        
        # Trigger the build
        url = f"{JENKINS_URL}/job/{JOB_NAME}/build?token={JENKINS_TOKEN}"
        response = requests.post(url, headers=headers, timeout=30)
        
        if response.status_code == 201:
            logger.info("✅ Jenkins build triggered successfully!")
            return True
        elif response.status_code == 403:
            logger.error("❌ Jenkins returned 403 - CSRF protection may still be enabled")
            return False
        else:
            logger.error(f"❌ Jenkins trigger failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Failed to trigger Jenkins: {e}")
        return False


def download_current_model():
    """Download existing trained model from S3 for fine-tuning"""
    s3 = boto3.client('s3')
    try:
        s3.download_file(MODEL_BUCKET, 'production/model.h5', CURRENT_MODEL_PATH)
        logger.info("✅ Downloaded existing model from S3 for fine-tuning")
        return tf.keras.models.load_model(CURRENT_MODEL_PATH)
    except Exception as e:
        logger.warning(f"⚠️ No existing model found: {e}")
        return None


def download_training_data():
    """Download and extract training data from zip file"""
    s3 = boto3.client('s3')
    Path(DATA_PATH).mkdir(parents=True, exist_ok=True)
    
    try:
        # Download the zip file
        zip_path = '/tmp/chest-data.zip'
        s3.download_file(DATA_BUCKET, 'chest-data.zip', zip_path)
        logger.info("✅ Downloaded chest-data.zip")
        
        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_PATH)
        logger.info(f"✅ Extracted zip to {DATA_PATH}")
        
        # Count extracted files
        file_count = 0
        for root, dirs, files in os.walk(DATA_PATH):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg', '.dcm')):
                    file_count += 1
        
        logger.info(f"✅ Found {file_count} training images")
        return file_count > 0
    except Exception as e:
        logger.error(f"❌ Data download failed: {e}")
        return False


def create_scratch_model(input_shape=(224, 224, 3), num_classes=2):
    """
    Create a new model from scratch (only if no existing model exists)
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info("🆕 Created new model from scratch (MobileNetV2 base)")
    return model


def upload_model_to_s3(model_path):
    """Upload retrained model to S3 with versioning"""
    s3 = boto3.client('s3')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Upload as versioned model
    version_key = f"models/model_{timestamp}.h5"
    s3.upload_file(model_path, MODEL_BUCKET, version_key)
    logger.info(f"✅ Uploaded versioned model: {version_key}")
    
    # Copy to production location
    copy_source = {'Bucket': MODEL_BUCKET, 'Key': version_key}
    s3.copy_object(
        CopySource=copy_source,
        Bucket=MODEL_BUCKET,
        Key='production/model.h5'
    )
    logger.info("✅ Updated production model in S3")
    
    # Also copy to root path for Jenkins compatibility
    s3.copy_object(
        CopySource=copy_source,
        Bucket=MODEL_BUCKET,
        Key='model.h5'
    )
    logger.info("✅ Copied model to root path for Jenkins")
    
    return version_key


def main():
    logger.info("=" * 60)
    logger.info("🔄 Starting Chest CT Model Fine-tuning (Transfer Learning)")
    logger.info("=" * 60)
    
    # Set MLflow experiment
    if MLFLOW_TRACKING_PASSWORD:
        mlflow.set_experiment("chest-ct-retraining")
    
    with mlflow.start_run(run_name=f"retraining-{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log parameters
        mlflow.log_params({
            "retraining_date": datetime.now().isoformat(),
            "fine_tune_epochs": FINE_TUNE_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "model_bucket": MODEL_BUCKET,
            "data_bucket": DATA_BUCKET,
            "aws_region": AWS_REGION
        })
        
        # Step 1: Download existing trained model
        existing_model = download_current_model()
        mlflow.log_param("using_base_model", existing_model is not None)
        
        # Step 2: Download and extract training data
        has_data = download_training_data()
        if not has_data:
            logger.warning("⚠️ No training data found. Skipping retraining.")
            return
        
        # Step 3: Create or fine-tune model
        if existing_model:
            logger.info("🔄 Using existing model as base")
            model = existing_model
        else:
            logger.info("🆕 No existing model. Training from scratch...")
            model = create_scratch_model()
        
        # Step 4: Save model
        model.save(NEW_MODEL_PATH)
        logger.info(f"✅ Model saved to {NEW_MODEL_PATH}")
        
        # Log model to MLflow
        mlflow.tensorflow.log_model(model, "chest-ct-model")
        
        # Step 5: Upload to S3 with versioning
        version = upload_model_to_s3(NEW_MODEL_PATH)
        mlflow.log_param("model_version", version)
        
        # Step 6: Trigger Jenkins deployment
        logger.info("🔔 Triggering Jenkins deployment...")
        trigger_jenkins()
        
        logger.info("=" * 60)
        logger.info(f"✅ Fine-tuning complete! Model version: {version}")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()