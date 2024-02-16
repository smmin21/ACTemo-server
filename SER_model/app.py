from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import io
from test import run_test

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...), emotion: str = Form(...)):
    # Read the audio file
    audio_data = await file.read()

    # Run the test
    predicted_emotion, level = run_test(io.BytesIO(audio_data), emotion)
    print(f'Predicted emotion in API: {predicted_emotion}, Level: {level}')

    # Return the response
    return JSONResponse(content={"level": level})

