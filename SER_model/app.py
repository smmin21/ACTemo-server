from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import io
from test import run_test

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the audio file
    audio_data = await file.read()

    # Run the test
    response = run_test(io.BytesIO(audio_data))
    print(f'Predicted emotion in API: {response}')

    # Return the response
    return JSONResponse(content={"emotion": response})

