from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os

app = FastAPI()

MODEL_PATH = os.environ.get("MODEL_PATH", os.path.expanduser("~/models/model.pkl"))

model = joblib.load(MODEL_PATH)
print(f"Model loaded from {MODEL_PATH}")


class PredictRequest(BaseModel):
    features: list[float]


LABEL_MAP = {0: "thap", 1: "trung_binh", 2: "cao"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    if len(req.features) != 12:
        raise HTTPException(
            status_code=400,
            detail="Expected 12 features (wine quality)",
        )

    pred = int(model.predict([req.features])[0])
    return {"prediction": pred, "label": LABEL_MAP[pred]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
