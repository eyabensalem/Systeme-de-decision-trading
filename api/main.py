from pathlib import Path
from fastapi import FastAPI
from contextlib import asynccontextmanager

from api.services.inference_service import InferenceService
from api.routers.health import router as health_router
from api.routers.model_info import router as info_router
from api.routers.predict import router as predict_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code exécuté au démarrage
    root = Path(__file__).resolve().parent.parent
    app.state.infer = InferenceService(root)

    yield  # l'application tourne ici

    # Code exécuté à l'arrêt (si besoin)
    # ex: fermer connexions, logs, etc.


app = FastAPI(
    title="Trading Decision API",
    lifespan=lifespan
)

@app.get("/")
def root():
    return {"message": "API is running. Go to /docs or /health"}


app.include_router(health_router)
app.include_router(info_router)
app.include_router(predict_router)
