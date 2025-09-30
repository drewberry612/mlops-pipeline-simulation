import json
import os
import io
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

from hyperparams import BATCH_SIZE, EPOCHS, LEARNING_RATE, OPTIMIZER, LOSS_FN, METRICS, IMG_SIZE, IMG_CHANNELS, TEST_SIZE, VAL_SIZE, RANDOM_STATE

def load_data():
	conn = sqlite3.connect('../data_warehouse/warehouse.db')

	df = pd.read_sql_query("""
		SELECT * FROM images
		WHERE processed = 1 AND label IS NOT NULL
	""", conn)

	image_dir = os.path.join(os.path.dirname(__file__), '../object_storage')

	images = []
	labels = []
	for _, row in df.iterrows():
		img_path = os.path.join(image_dir, row['filename'])
		if os.path.exists(img_path):
			img = Image.open(img_path)
			images.append(np.array(img))
			labels.append(int(row['label']))

	images = np.array(images)
	labels = np.array(labels)

	conn.close()

	return images, labels

def preprocess_data(images, labels):
	images = images / 255.0
	images = images.reshape(-1, IMG_SIZE[0], IMG_SIZE[1], IMG_CHANNELS)

	X_temp, X_test, y_temp, y_test = train_test_split(images, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE)
	X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=VAL_SIZE, random_state=RANDOM_STATE)
	
	return X_train, X_val, X_test, y_train, y_val, y_test

def build_model(num_classes):
	model = models.Sequential()
	model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], IMG_CHANNELS)))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Conv2D(64, (3, 3), activation='relu'))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Conv2D(64, (3, 3), activation='relu'))
	model.add(layers.Flatten())
	model.add(layers.Dense(64, activation='relu'))
	model.add(layers.Dense(num_classes))

	return model

def train_model(model, X_train, y_train, X_val, y_val):
	model.compile(
		optimizer=OPTIMIZER,
		loss=LOSS_FN,
		metrics=METRICS
	)

	history = model.fit(
		X_train, y_train,
		epochs=EPOCHS,
		batch_size=BATCH_SIZE,
		validation_data=(X_val, y_val)
	)

	return history

def evaluate_model(model, X_test, y_test):
	test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
	return test_loss, test_acc

def save_model(model):
	save_dir = os.path.join(os.path.dirname(__file__), '../model_registry')
	os.makedirs(save_dir, exist_ok=True)

	existing = [d for d in os.listdir(save_dir) if d.startswith('v') and d[1:].isdigit()]
	if existing:
		next_num = max([int(d[1:]) for d in existing]) + 1
	else:
		next_num = 1

	model_path = os.path.join(save_dir, f'v{next_num}')
	model.save(model_path)

	return model_path

def log_training_run(training_time, test_acc, test_loss, history, model_path, model_summary_str):
	conn = sqlite3.connect('../data_warehouse/warehouse.db')

	history_json = json.dumps(history.history)

	hyperparams_dict = {
		'BATCH_SIZE': BATCH_SIZE,
		'EPOCHS': EPOCHS,
		'LEARNING_RATE': LEARNING_RATE,
		'OPTIMIZER': OPTIMIZER,
		'LOSS_FN': str(LOSS_FN),
		'METRICS': METRICS,
		'IMG_SIZE': IMG_SIZE,
		'IMG_CHANNELS': IMG_CHANNELS,
		'TEST_SIZE': TEST_SIZE,
		'VAL_SIZE': VAL_SIZE,
		'RANDOM_STATE': RANDOM_STATE
	}
	hyperparams_json = json.dumps(hyperparams_dict)

	# Save model summary as text file in model directory
	summary_path = os.path.join(model_path, 'model_summary.txt')
	with open(summary_path, 'w', encoding='utf-8') as f:
		f.write(model_summary_str)

	cursor = conn.cursor()
	cursor.execute(
		"""
		INSERT INTO training_runs (training_time, accuracy, loss, history, model_path, hyperparams)
		VALUES (?, ?, ?, ?, ?, ?)
		""",
		(training_time, float(test_acc), float(test_loss), history_json, os.path.abspath(model_path), hyperparams_json)
	)
	conn.commit()
	cursor.close()
	conn.close()

def run_training_pipeline():
    # Load and preprocess data
    images, labels = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(images, labels)

    # Build model
    num_classes = len(np.unique(labels))
    model = build_model(num_classes)

    # Capture model summary
    model_summary_io = io.StringIO()
    model.summary(print_fn=lambda x: model_summary_io.write(x + '\n'))
    model_summary_str = model_summary_io.getvalue()
    model_summary_io.close()

    # Train model
    train_start_time = datetime.now()
    history = train_model(model, X_train, y_train, X_val, y_val)
    train_end_time = datetime.now()
    training_time = (train_end_time - train_start_time).total_seconds()

    # Evaluate model
    test_loss, test_acc = evaluate_model(model, X_test, y_test)

    # Save model
    model_path = save_model(model)

    # Log training run
    log_training_run(training_time, test_acc, test_loss, history, model_path, model_summary_str)

    # Return results
    return {
        "model_path": model_path,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "training_time": training_time,
        "model_summary": model_summary_str,
        "history": history.history
    }

# Example usage for testing (remove or comment out for API use)
if __name__ == "__main__":
	result = run_training_pipeline()
	print(json.dumps(result, indent=2))
