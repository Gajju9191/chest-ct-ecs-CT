#!/usr/bin/env python3
"""
AWS Batch Retraining Script - Complete Working Version with Jenkins Auth
"""

import boto3
import tensorflow as tf
import mlflow
import mlflow.tensorflow
import os
import requests
import json
import logging
from datetime import datetime
from requests.auth import HTTPBasicAuth

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration - Get from environment variables
MODEL_BUCKET = os.environ.get('MODEL_BUCKET', 'chest-ct-models-155407238003')
DATA_BUCKET = os.environ.get('DATA_BUCKET', 'chest-ct-raw-data')
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')
JENKINS_URL = os.environ.get('JENKINS_URL', 'http://100.51.185.244:8080')
JENKINS_TOKEN = os.environ.get('JENKINS_TOKEN', 'ct-trigger-token')
JENKINS_USERNAME = os.environ.get('JENKINS_USERNAME', 'Gajanan Wagalgave')
JENKINS_API_TOKEN = os.environ.get('JENKINS_API_TOKEN', '')  # Set this in Batch environment
JOB_NAME = "Gajanan Wagalgave"

# Local paths
CURRENT_MODEL_PATH = '/tmp/current_model.h5'
NEW_MODEL_PATH = '/tmp/new_model.h5'
DATA_PATH = '/tmp/data/'
BATCH_SIZE = 32
EPOCHS = 5


def trigger_jenkins():
    """Trigger Jenkins deployment with authentication and CSRF crumb"""
    try:
        url = f"{JENKINS_URL}/job/{JOB_NAME}/build?token={JENKINS_TOKEN}"
        auth = HTTPBasicAuth(JENKINS_USERNAME, JENKINS_API_TOKEN)
        
        logger.info(f"🔔 Triggering Jenkins at: {url}")
        
        # Get crumb for CSRF protection
        crumb_url = f"{JENKINS_URL}/crumbIssuer/api/json"
        crumb_response = requests.get(crumb_url, auth=auth, timeout=30)
        
        if crumb_response.status_code == 200:
            crumb = crumb_response.json()['crumb']
            crumb_field = crumb_response.json()['crumbRequestField']
            headers = {crumb_field: crumb}
            logger.info("✅ CSRF crumb obtained")
        else:
            headers = {}
            logger.warning(f"Could not fetch crumb: {crumb_response.status_code}")
        
        # Trigger the build
        response = requests.post(url, headers=headers, auth=auth, timeout=30)
        
        if response.status_code == 201:
            logger.info("✅ Jenkins build triggered successfully!")
            return True
        else:
            logger.error(f"❌ Jenkins trigger failed: {response.status_code}")
            logger.error(f"Response: {response.text[:500]}")
            return False
    except Exception as e:
        logger.error(f"❌ Failed to trigger Jenkins: {e}")
        return False


def download_current_model():
    """Download existing model from S3"""
    s3 = boto3.client('s3')
    try:
        s3.download_file(MODEL_BUCKET, 'production/model.h5', CURRENT_MODEL_PATH)
        logger.info("✅ Downloaded existing model from S3")
        return tf.keras.models.load_model(CURRENT_MODEL_PATH)
    except Exception as e:
        logger.warning(f"⚠️ No existing model found: {e}")
        return None


def download_training_data():
    """Download training data from S3"""
    s3 = boto3.client('s3')
    os.makedirs(DATA_PATH, exist_ok=True)
    
    try:
        response = s3.list_objects_v2(Bucket=DATA_BUCKET, Prefix='training/')
        
        if 'Contents' not in response:
            logger.warning("No training data found in S3")
            return False
        
        file_count = 0
        for obj in response['Contents']:
            key = obj['Key']
            local_file = os.path.join(DATA_PATH, os.path.basename(key))
            s3.download_file(DATA_BUCKET, key, local_file)
            file_count += 1
        
        logger.info(f"✅ Downloaded {file_count} training files")
        return file_count > 0
    except Exception as e:
        logger.error(f"❌ Data download failed: {e}")
        return False


def create_model():
    """Create a simple CNN model for chest CT classification"""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def upload_model_to_s3(model_path):
    """Upload retrained model to S3"""
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
    
    return version_key


def main():
    logger.info("=" * 60)
    logger.info("🔄 Starting Chest CT Model Retraining")
    logger.info("=" * 60)
    
    # Step 1: Download existing model
    current_model = download_current_model()
    if current_model:
        logger.info("✅ Current model loaded")
    
    # Step 2: Download new training data
    has_data = download_training_data()
    if not has_data:
        logger.warning("⚠️ No new data found. Skipping retraining.")
        return
    
    # Step 3: Create new model
    logger.info("🏗️ Creating new model...")
    model = create_model()
    
    # Step 4: Save model
    model.save(NEW_MODEL_PATH)
    logger.info(f"✅ Model saved to {NEW_MODEL_PATH}")
    
    # Step 5: Upload to S3
    version = upload_model_to_s3(NEW_MODEL_PATH)
    
    # Step 6: Trigger Jenkins deployment
    logger.info("🔔 Triggering Jenkins deployment...")
    trigger_jenkins()
    
    logger.info("=" * 60)
    logger.info(f"✅ Retraining complete! Model version: {version}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()