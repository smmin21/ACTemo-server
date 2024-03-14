# from fastapi.testclient import TestClient
from app import app
from io import BytesIO
import httpx
from fastapi.testclient import TestClient

client = TestClient(app)

def test_predict():
    audio_file = BytesIO(open("test_data/angry1.wav", "rb").read())

    response = client.post(
        "/predict",
        files={"file": ("angry1.wav", audio_file, "audio/wav")},
        data={"emotion": "Angry"}
    )
    assert response.status_code == 200
    assert response.json() == {"level": "A"}