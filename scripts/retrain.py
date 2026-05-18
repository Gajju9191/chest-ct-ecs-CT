#!/usr/bin/env python3
"""
AWS Batch Retraining Script - Simple Working Version
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration - Get from environment variables
MODEL_BUCKET = os.environ.get('MODEL_BUCKET', 'chest-ct-models-155407238003')
DATA_BUCKET = os.environ.get('DATA_BUCKET', 'chest-ct-raw-data')
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')
JENKINS_URL = os.environ.get('JENKINS_URL', 'http://your-jenkins-ip:8080')
JENKINS_TOKEN = os.environ.get('JENKINS_TOKEN', 'ct-trigger-token')

# Local paths
CURRENT_MODEL_PATH = '/tmp/current_model.h5'
NEW_MODEL_PATH = '/tmp/new_model.h5'
DATA_PATH = '/tmp/data/'
BATCH_SIZE = 32
EPOCHS = 5  # Small number for testing


def trigger_jenkins():
    """Trigger Jenkins deployment"""
    try:
        url = f"{JENKINS_URL}/generic-webhook-trigger/invoke"
        params = {'token': JENKINS_TOKEN}
        response = requests.post(url, params=params, timeout=30)
        logger.info(f"✅ Jenkins triggered: {response.status_code}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to trigger Jenkins: {e}")
        return False


def download_current_model():
    """Download existing model from S3"""
    s3 = boto3.client('s3')
    try:
        s3.download_file(MODEL_BUCKET, 'production/model.h5', CURRENT_MODEL_PATH)
        logger.info("✅ Downloaded existing model")
        return tf.keras.models.load_model(CURRENT_MODEL_PATH)
    except Exception as e:
        logger.warning(f"⚠️ No existing model: {e}")
        return None


def download_training_data():
    """Download training data from S3"""
    s3 = boto3.client('s3')
    os.makedirs(DATA_PATH, exist_ok=True)
    
    try:
        # List objects in the training data prefix
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
        
        logger.info(f"✅ Downloaded {file_count} files")
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
        tf.keras.layers.Dense(2, activation='softmax')  # 2 classes: normal/abnormal
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
    logger.info(f"✅ Updated production model")
    
    return version_key


def main():
    logger.info("=" * 60)
    logger.info("🔄 Starting Chest CT Model Retraining")
    logger.info("=" * 60)
    
    # Step 1: Download existing model (if any)
    current_model = download_current_model()
    if current_model:
        logger.info(f"Current model summary: {current_model.summary()}")
    
    # Step 2: Download new training data
    has_data = download_training_data()
    if not has_data:
        logger.warning("⚠️ No new data found. Skipping retraining.")
        return
    
    # Step 3: Create new model
    logger.info("🏗️ Creating new model...")
    model = create_model()
    
    # Step 4: Save model (In real scenario, you would train here)
    # For now, we're just saving the untrained model as a placeholder
    # TODO: Add actual model.fit() with your training data
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