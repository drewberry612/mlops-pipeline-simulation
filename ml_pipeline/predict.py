import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from hyperparams import IMG_SIZE, IMG_CHANNELS
from class_names import class_names
from preprocess import preprocess_image

def load_latest_model():
    model_dir = os.path.join(os.path.dirname(__file__), '../model_registry')

    existing = [d for d in os.listdir(model_dir) if d.startswith('v') and d[1:].isdigit()]
    if not existing:
        raise FileNotFoundError('No saved models found in model_registry.')
    
    latest_num = max([int(d[1:]) for d in existing])
    model_path = os.path.join(model_dir, f'v{latest_num}')

    return load_model(model_path)

def get_top3_predictions(model, input_image):
    predicted_probs = model.predict(input_image)

    top3_indices = np.argsort(predicted_probs, axis=1)[:, -3:][:, ::-1][0]
    top3_probs = predicted_probs[0, top3_indices]
    results = [(class_names[i], float(p)) for i, p in zip(top3_indices, top3_probs)]

    return results, top3_probs[0]

def predict_image(image_array=None, threshold=0.5):
    """
    Main API function. Returns top-3 predictions for an image array.
    If image_array is None, uses a random image for testing.
    """
    model = load_latest_model()

    input_image = preprocess_image() # change to not open the file

    results, top1_prob = get_top3_predictions(model, input_image)

    if top1_prob < threshold:
        return {"message": "Image could not be classified confidently.", "results": results}
    else:
        return {"message": "Top-3 predictions:", "results": results}

# Example usage for testing (remove or comment out for API use)
if __name__ == "__main__":
    output = predict_image()
    print(output["message"])
    for label, prob in output["results"]:
        print(f"  {label}: {prob:.2f}")