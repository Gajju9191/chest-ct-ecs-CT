#!/usr/bin/env python3
"""
AWS Batch Retraining Script - Proper Fine-tuning with Transfer Learning
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
from datetime import datetime
from pathlib import Path
from requests.auth import HTTPBasicAuth

# Robust import for ImageDataGenerator (works in different environments)
try:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    logger_import = logging.getLogger(__name__)
    logger_import.info("Using ImageDataGenerator from tensorflow.keras")
except ImportError:
    try:
        from keras.preprocessing.image import ImageDataGenerator
        logger_import = logging.getLogger(__name__)
        logger_import.info("Using ImageDataGenerator from keras.preprocessing")
    except ImportError:
        from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
        logger_import = logging.getLogger(__name__)
        logger_import.info("Using ImageDataGenerator from tensorflow.python.keras")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration - Get from environment variables
MODEL_BUCKET = os.environ.get('MODEL_BUCKET', 'chest-ct-models-155407238003')
DATA_BUCKET = os.environ.get('DATA_BUCKET', 'chest-ct-raw-data')
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')
JENKINS_URL = os.environ.get('JENKINS_URL', 'http://100.51.185.244:8080')
JENKINS_TOKEN = os.environ.get('JENKINS_TOKEN', 'ct-trigger-token')
JOB_NAME = "Gajanan Wagalgave"

# Local paths
CURRENT_MODEL_PATH = '/tmp/current_model.h5'
NEW_MODEL_PATH = '/tmp/new_model.h5'
DATA_PATH = '/tmp/data/'
BATCH_SIZE = 32
FINE_TUNE_EPOCHS = 10  # Few epochs for fine-tuning
LEARNING_RATE = 1e-5   # Lower learning rate for fine-tuning


def trigger_jenkins():
    """Trigger Jenkins deployment using remote build trigger"""
    try:
        url = f"{JENKINS_URL}/job/{JOB_NAME}/build?token={JENKINS_TOKEN}"
        response = requests.post(url, timeout=30)
        
        if response.status_code == 201:
            logger.info("✅ Jenkins build triggered successfully!")
            return True
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
    """Download new chest CT data from S3"""
    s3 = boto3.client('s3')
    Path(DATA_PATH).mkdir(parents=True, exist_ok=True)
    
    try:
        # List objects in the training data prefix
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=DATA_BUCKET, Prefix='training/')
        
        file_count = 0
        for page in pages:
            if 'Contents' not in page:
                continue
            for obj in page['Contents']:
                key = obj['Key']
                local_file = Path(DATA_PATH) / key.replace('training/', '')
                local_file.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(DATA_BUCKET, key, str(local_file))
                file_count += 1
        
        logger.info(f"✅ Downloaded {file_count} new files for fine-tuning")
        return file_count > 0
    except Exception as e:
        logger.error(f"❌ Data download failed: {e}")
        return False


def load_data_generators(data_path, img_size=(224, 224)):
    """Load and preprocess image data from directory structure"""
    # Data augmentation for fine-tuning (light augmentation)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    # Load training data
    train_generator = train_datagen.flow_from_directory(
        data_path,
        target_size=img_size,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )
    
    # Load validation data
    val_generator = train_datagen.flow_from_directory(
        data_path,
        target_size=img_size,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )
    
    logger.info(f"✅ Loaded {train_generator.samples} training images, {val_generator.samples} validation images")
    
    return train_generator, val_generator


def fine_tune_model(model, train_generator, val_generator):
    """
    Fine-tune existing model with new data using transfer learning
    """
    # Count trainable layers before
    trainable_before = sum(layer.trainable for layer in model.layers)
    
    # Freeze early layers (preserve learned features)
    # Keep only the last 20 layers trainable
    for layer in model.layers[:-20]:
        layer.trainable = False
    
    # Recompile with lower learning rate for fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    trainable_after = sum(layer.trainable for layer in model.layers)
    logger.info(f"🔄 Fine-tuning: {trainable_before} → {trainable_after} trainable layers")
    logger.info(f"📉 Learning rate reduced to: {LEARNING_RATE}")
    
    # Fine-tune on new data
    logger.info(f"🚀 Starting fine-tuning for {FINE_TUNE_EPOCHS} epochs...")
    
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=FINE_TUNE_EPOCHS,
        verbose=1
    )
    
    # Log metrics
    final_accuracy = history.history['accuracy'][-1]
    final_val_accuracy = history.history['val_accuracy'][-1]
    
    logger.info(f"📊 Fine-tuning Results:")
    logger.info(f"   - Training Accuracy: {final_accuracy:.4f}")
    logger.info(f"   - Validation Accuracy: {final_val_accuracy:.4f}")
    
    return model, history


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
    
    return version_key


def main():
    logger.info("=" * 60)
    logger.info("🔄 Starting Chest CT Model Fine-tuning (Transfer Learning)")
    logger.info("=" * 60)
    
    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_params({
            "retraining_date": datetime.now().isoformat(),
            "fine_tune_epochs": FINE_TUNE_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE
        })
        
        # Step 1: Download existing trained model
        existing_model = download_current_model()
        
        # Step 2: Download new training data
        has_data = download_training_data()
        if not has_data:
            logger.warning("⚠️ No new data found. Skipping retraining.")
            return
        
        # Step 3: Load data generators
        train_generator, val_generator = load_data_generators(DATA_PATH)
        
        # Step 4: Fine-tune or train from scratch
        if existing_model:
            logger.info("🔄 Fine-tuning existing model with new data...")
            model, history = fine_tune_model(existing_model, train_generator, val_generator)
            mlflow.log_param("training_type", "fine_tune")
            
            # Log final metrics
            mlflow.log_metrics({
                "final_accuracy": history.history['accuracy'][-1],
                "final_val_accuracy": history.history['val_accuracy'][-1]
            })
        else:
            logger.info("🆕 No existing model. Training from scratch...")
            model = create_scratch_model()
            
            # Train from scratch (more epochs)
            logger.info("🚀 Training from scratch...")
            history = model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=30,
                verbose=1
            )
            mlflow.log_param("training_type", "from_scratch")
        
        # Step 5: Save model
        model.save(NEW_MODEL_PATH)
        logger.info(f"✅ Model saved to {NEW_MODEL_PATH}")
        
        # Step 6: Upload to S3 with versioning
        version = upload_model_to_s3(NEW_MODEL_PATH)
        
        # Step 7: Trigger Jenkins deployment
        logger.info("🔔 Triggering Jenkins deployment...")
        trigger_jenkins()
        
        logger.info("=" * 60)
        logger.info(f"✅ Fine-tuning complete! Model version: {version}")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()