# ===============================
# Imports and Setup
# ===============================

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from hyperparams import IMG_SIZE, IMG_CHANNELS

# ===============================
# Load Latest Model
# ===============================

model_dir = os.path.join(os.path.dirname(__file__), '../model_registry')
existing = [d for d in os.listdir(model_dir) if d.startswith('v') and d[1:].isdigit()]
if not existing:
    raise FileNotFoundError('No saved models found in model_registry.')
latest_num = max([int(d[1:]) for d in existing])
model_path = os.path.join(model_dir, f'v{latest_num}')
model = load_model(model_path)

# ===============================
# Placeholder for Input Data
# ===============================

# Replace this with actual image loading and preprocessing
# Example: single 28x28 grayscale image, normalized
input_image = np.random.rand(1, IMG_SIZE[0], IMG_SIZE[1], IMG_CHANNELS).astype(np.float32)

# ===============================
# Prediction
# ===============================

logits = model.predict(input_image)
predicted_class = np.argmax(logits, axis=1)
