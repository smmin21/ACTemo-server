from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import io
from test import run_test
import multiprocessing

app = FastAPI()

async def startup_event_handler():
    app.state.process_pool = multiprocessing.Pool(7)

async def shutdown_event_handler():
    app.state.process_pool.close()
    app.state.process_pool.join()

app.add_event_handler("startup", startup_event_handler)
app.add_event_handler("shutdown", shutdown_event_handler)

@app.post("/predict")
async def predict(file: UploadFile = File(...), emotion: str = Form(...)):
    # Read the audio file
    audio_data = await file.read()

    # Run the test
    predicted_emotion, level = run_test(io.BytesIO(audio_data), emotion, app.state.process_pool)
    print(f'Predicted emotion in API: {predicted_emotion}, Level: {level}')

    # Return the response
    return JSONResponse(content={"level": level})

