from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image
import io
import tensorflow as tf

app = FastAPI()

# Allow CORS for your React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Adjust this as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
model = tf.keras.models.load_model(r"C:\Users\arion\Desktop\dataset\plantvillage dataset\mltrain\plant_disease_model.h5")

# List of disease names corresponding to model output classes
disease_names = [
    "Apple Scab",                      # Class 0
    "Apple Black Rot",                 # Class 1
    "Cedar Apple Rust",                # Class 2
    "Healthy Apple",                   # Class 3
    "Healthy Blueberry",               # Class 4
    "Cherry Powdery Mildew",           # Class 5
    "Corn Cercospora Leaf Spot",       # Class 6
    "Corn Common Rust",                # Class 7
    "Corn Northern Leaf Blight",       # Class 8
    "Healthy Corn",                    # Class 9
    "Grape Black Rot",                 # Class 10
    "Grape Esca",                      # Class 11
    "Grape Leaf Blight",               # Class 12
    "Healthy Grape",                   # Class 13
    "Orange Citrus Greening",          # Class 14
    "Peach Bacterial Spot",            # Class 15
    "Healthy Peach",                   # Class 16
    "Pepper Bell Bacterial Spot",      # Class 17
    "Healthy Pepper Bell",             # Class 18
    "Potato Early Blight",             # Class 19
    "Potato Late Blight",              # Class 20
    "Healthy Potato",                  # Class 21
    "Tomato Bacterial Spot",           # Class 22
    "Tomato Early Blight",             # Class 23
    "Tomato Late Blight",              # Class 24
    "Tomato Leaf Mold",                # Class 25
    "Tomato Septoria Leaf Spot",       # Class 26
    "Tomato Spider Mite",              # Class 27
    "Tomato Target Spot",              # Class 28
    "Tomato Yellow Leaf Curl Virus"    # Class 29
]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).resize((224, 224))  # Resize based on your model input
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.expand_dims(image, axis=0)

    # Make prediction
    prediction = model.predict(image)
    predicted_class = prediction.argmax()

    # Get disease name from the list
    disease_name = disease_names[predicted_class]

    return JSONResponse(content={"prediction": disease_name})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
