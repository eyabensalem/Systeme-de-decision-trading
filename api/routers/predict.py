from fastapi import APIRouter, Request
from api.schemas.request import PredictRequest
from api.schemas.response import PredictResponse

router = APIRouter()

@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request):
    svc = request.app.state.infer
    action, score = svc.predict(req.features)
    return PredictResponse(
        action=action,
        score=score,
        model_type=svc.model_type,
        model_dir=str(svc.model_dir),
    )
