import tensorflow as tf

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
OPTIMIZER = 'adam'  # or tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
LOSS_FN = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
METRICS = ['accuracy']
IMG_SIZE = (28, 28)
IMG_CHANNELS = 1
TEST_SIZE = 0.2
VAL_SIZE = 0.25
RANDOM_STATE = 42
