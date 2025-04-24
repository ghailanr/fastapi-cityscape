import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from models.unet import load_model, predict_mask
from contextlib import asynccontextmanager


models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Chargement du modèle...")
    models['Unet'] = load_model()
    print("Modèle chargé.")
    yield
    print("Unloads model")
    models.clear()


app = FastAPI(lifespan=lifespan)


@app.post("/predict/")
async def predict(image: UploadFile = File(...)):
    result_img_bytes = predict_mask(image, models['Unet'])
    return Response(content=result_img_bytes, media_type="image/png")


@app.get("/health")
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000)
