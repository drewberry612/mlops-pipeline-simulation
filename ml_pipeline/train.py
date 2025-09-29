# ===============================
# Imports and Setup
# ===============================

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

# ===============================
# Data Loading
# ===============================

conn = sqlite3.connect('../data_warehouse/warehouse.db')

df = pd.read_sql_query("""
	SELECT * FROM images
	WHERE processed = 1 AND label IS NOT NULL
""", conn)

def load_image(path):
	img = Image.open(path).convert('L')  # grayscale
	img = img.resize(IMG_SIZE)
	return np.array(img)

image_dir = os.path.join(os.path.dirname(__file__), '../object_storage')
images = []
labels = []
for _, row in df.iterrows():
	img_path = os.path.join(image_dir, row['filename'])
	if os.path.exists(img_path):
		images.append(load_image(img_path))
		labels.append(int(row['label']))

images = np.array(images)
labels = np.array(labels)

# ===============================
# Data Preprocessing
# ===============================

# Normalize and reshape
images = images / 255.0
images = images.reshape(-1, IMG_SIZE[0], IMG_SIZE[1], IMG_CHANNELS)

# Split into train, validation, and test sets
X_temp, X_test, y_temp, y_test = train_test_split(images, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=VAL_SIZE, random_state=RANDOM_STATE)  # 0.25 x 0.8 = 0.2

# ===============================
# Model Definition
# ===============================

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], IMG_CHANNELS)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10)) # change this for number of classes

# ===============================
# Model Summary Capture
# ===============================

model_summary_io = io.StringIO()
model.summary(print_fn=lambda x: model_summary_io.write(x + '\n'))
model_summary_str = model_summary_io.getvalue()
model_summary_io.close()

# ===============================
# Training
# ===============================

train_start_time = datetime.now()

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

train_end_time = datetime.now()

# ===============================
# Evaluation
# ===============================

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

# ===============================
# Model Saving
# ===============================

save_dir = os.path.join(os.path.dirname(__file__), '../model_registry')
os.makedirs(save_dir, exist_ok=True)
existing = [d for d in os.listdir(save_dir) if d.startswith('v') and d[1:].isdigit()]
if existing:
	next_num = max([int(d[1:]) for d in existing]) + 1
else:
	next_num = 1
model_path = os.path.join(save_dir, f'v{next_num}')

model.save(model_path)

# ===============================
# Log Training Run
# ===============================

training_time = (train_end_time - train_start_time).total_seconds()
history_json = json.dumps(history.history)

# Serialize hyperparameters for logging
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
