from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
from  io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()


prod_model = tf.keras.models.load_model("../saved_models/1.h5")
beta_model = tf.keras.models.load_model("../saved_models/2.h5")

MODEL = tf.keras.models.load_model("../saved_models/1.h5")
CLASS_NAMES = ["EarlyBlight", "Late Blight", "Healthy"]



@app.get("/ping")
async def ping():
    return "Helllo, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
    ):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)

    predicted_class= CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }
if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)

