from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ml_pipeline.predict import predict_image
import numpy as np
from PIL import Image
import io

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
	try:
		contents = await file.read()
		image = Image.open(io.BytesIO(contents)).convert("RGB")
		image_array = np.array(image)
		result = predict_image(image_array=image_array)
		# Format output for client
		predictions = [
			{"label": label, "probability": prob}
			for label, prob in result["results"]
		]
		return JSONResponse(content={"predictions": predictions, "message": result["message"]})
	except Exception as e:
		return JSONResponse(content={"error": str(e)}, status_code=500)
